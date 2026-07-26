/// JIT Source Map — VmInstr → 机器码偏移 → Op 标签映射。
/// 仅当 CompileConfig.debug_jit=true 时生成。
/// DAP 调试器用它做 source → address 映射。
//
// @trace REQ-DUMP-003 [entity:ENT-COMPILER-GRAPH] VmInstr 序列完整 codegen 结果 dump —
// offset map 扩展了 dump 能力: 除 VmInstr 序列本身外, 现在还提供 VmInstr →
// 机器码字节偏移的映射, 以及 const_pool/data_tables 布局审计, 用于不依赖 GDB
// 的静态诊断 (BCE-20260724-PLAN-C-RESIDUAL-BREAK: Q5_K_M N=28 SIGSEGV 定位).

/// JIT source map 条目集合。
#[derive(Debug, Clone, Default)]
pub struct JitSourceMap {
    pub entries: Vec<JitSourceEntry>,
}

/// 单条映射：机器码偏移 → 源信息。
#[derive(Debug, Clone)]
pub struct JitSourceEntry {
    /// 该 VmInstr 在 JIT 机器码中的字节偏移
    pub code_offset: u32,
    /// 对应的源信息
    pub source: SourceInfo,
}

/// 源信息 — 描述此位置在 mega-kernel 中的逻辑角色。
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// CompilerGraph 中的 Op 名称（如 "L0.q_proj", "embed"）
    pub op_label: Option<String>,
    /// 融合组 ID
    pub fusion_group: Option<usize>,
    /// mega-kernel 阶段
    pub phase: String,
    /// DebugBreakpoint label 或 DebugMarker message
    pub debug_label: Option<String>,
}

impl JitSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录一条 debug 条目。
    pub fn add(&mut self, code_offset: u32, phase: &str, debug_label: String) {
        self.entries.push(JitSourceEntry {
            code_offset,
            source: SourceInfo {
                op_label: None,
                fusion_group: None,
                phase: phase.to_string(),
                debug_label: Some(debug_label),
            },
        });
    }

    /// 按 code_offset 排序（lowering 后调用一次）。
    pub fn sort_by_offset(&mut self) {
        self.entries.sort_by_key(|e| e.code_offset);
    }

    /// 序列化为人类可读的文本格式（无需 serde）。
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(self.entries.len() * 80);
        for e in &self.entries {
            let label = e.source.debug_label.as_deref().unwrap_or("?");
            let phase = &e.source.phase;
            let op = e.source.op_label.as_deref().unwrap_or("-");
            out.push_str(&format!("0x{:06x}  [{:>12}]  {}  ({})\n", e.code_offset, phase, label, op));
        }
        out
    }
}

// ── VmInstr → 机器码字节偏移映射 (BCE-20260724-PLAN-C-RESIDUAL-BREAK) ──
//
// 诊断工具: 不依赖 GDB 定位 SIGSEGV 崩溃 VmInstr。JIT 代码 + VmInstr 序列 +
// 此 map → 静态读 VmInstr dump + 偏移即可定位崩溃机器码偏移对应的 VmInstr。
//
// @trace REQ-DUMP-003 [entity:ENT-COMPILER-GRAPH] VmInstr 序列完整 codegen 结果 dump 扩展:
// 除 VmInstr 序列本身外, 提供 VmInstr → 机器码字节偏移区间映射, 用于无 GDB 静态诊断。

/// 单条 VmInstr 偏移映射: 该 VmInstr 对应的机器码字节区间。
///
/// `start_byte_off..end_byte_off` 是该 VmInstr lowering 产出的所有 iced 指令
/// assemble 后的字节偏移区间 (半开区间, 基于 base IP=0x0)。
#[derive(Debug, Clone)]
pub struct VmInstrOffsetEntry {
    /// VmInstr 在 VmProgram.instrs 中的索引。
    pub vm_instr_index: usize,
    /// 该 VmInstr 第一条机器指令的字节偏移 (assemble 后)。
    pub start_byte_off: u32,
    /// 该 VmInstr 最后一条机器指令的下一字节偏移 (半开区间)。
    pub end_byte_off: u32,
    /// VmInstr 的 Debug 摘要 (变体名, 如 "VecLoad", "Q5KDecodeStep")。
    pub instr_debug: String,
}

/// VmInstr → 机器码字节偏移映射集合。
///
/// 由 `X86Lower::finalize` 在 `assemble_options(RETURN_NEW_INSTRUCTION_OFFSETS)` 后构建:
/// 遍历 `vm_instr_offsets` (lower_instr 成功路径记录的 iced 指令索引区间),
/// 用 `new_instruction_offsets` 把索引转成字节偏移。
#[derive(Debug, Clone, Default)]
pub struct VmInstrOffsetMap {
    pub entries: Vec<VmInstrOffsetEntry>,
}

impl VmInstrOffsetMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// 按字节偏移排序 (finalize 后调用一次)。
    pub fn sort_by_offset(&mut self) {
        self.entries.sort_by_key(|e| e.start_byte_off);
    }

    /// 二分查找: 给定机器码字节偏移, 返回包含它的 VmInstr 索引 (None 表示未命中)。
    ///
    /// 用于崩溃诊断: 已知崩溃 RIP 相对偏移 → 查此 map → 定位崩溃 VmInstr。
    pub fn lookup_byte_offset(&self, byte_off: u32) -> Option<usize> {
        // entries 按 start_byte_off 排序, 二分找最后一个 start_byte_off <= byte_off 的 entry。
        let idx = self.entries.partition_point(|e| e.start_byte_off <= byte_off);
        if idx == 0 {
            return None;
        }
        let entry = &self.entries[idx - 1];
        if byte_off < entry.end_byte_off {
            Some(entry.vm_instr_index)
        } else {
            None
        }
    }

    /// 序列化为人类可读的文本格式 (诊断 dump 用)。
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(self.entries.len() * 96);
        for e in &self.entries {
            out.push_str(&format!(
                "VmInstr[{:4}] @ byte 0x{:06x}-0x{:06x} | {}\n",
                e.vm_instr_index, e.start_byte_off, e.end_byte_off, e.instr_debug
            ));
        }
        out
    }
}

// ── const_pool / data_tables 布局审计 (BCE-20260724-PLAN-C-RESIDUAL-BREAK) ──
//
// 诊断工具: 验证 "代码大小敏感的 spill slot 布局 + regalloc offset 管理" 假设。
// Q5 大代码 (177861 机器指令) 下 Plan C 的 150 条 AddPtr 改变 const_pool/data_tables
// 位置 → RIP-relative 偏移可能溢出/错位。此审计 dump 每个 entry 的 label/偏移/大小
// + 所有引用该 label 的 RIP-relative 指令的偏移 + disp32 值。

/// 单条 const_pool/data_tables 审计条目。
#[derive(Debug, Clone)]
pub struct PoolTableEntry {
    /// 表类型 ("const_pool" 或 "data_tables")。
    pub table_kind: String,
    /// entry 在表中的索引。
    pub entry_index: usize,
    /// entry 在机器码中的字节偏移 (assemble 后)。
    pub byte_offset: u32,
    /// entry 字节大小。
    pub size: u32,
}

/// 引用某个 const_pool/data_tables entry 的 RIP-relative 指令审计。
#[derive(Debug, Clone)]
pub struct RipRelRef {
    /// 引用指令的字节偏移。
    pub instr_byte_off: u32,
    /// 该指令编码的 disp32 值 (RIP-relative displacement)。
    pub disp32: i32,
    /// 引用目标的字节偏移 (instr_end_ip + disp32, 即指令下一字节 IP + disp)。
    pub target_byte_off: u64,
}

/// const_pool / data_tables 布局审计结果。
#[derive(Debug, Clone, Default)]
pub struct ConstPoolAudit {
    /// 所有 const_pool / data_tables entry 的布局信息。
    pub entries: Vec<PoolTableEntry>,
    /// 所有引用 const_pool / data_tables label 的 RIP-relative 指令。
    pub rip_refs: Vec<RipRelRef>,
}

impl ConstPoolAudit {
    pub fn new() -> Self {
        Self::default()
    }

    /// 序列化为人类可读的文本格式 (诊断 dump 用)。
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(512 + self.entries.len() * 64 + self.rip_refs.len() * 80);
        out.push_str(&format!("=== ConstPool/DataTables Audit ({} entries, {} rip-refs) ===\n",
            self.entries.len(), self.rip_refs.len()));
        for e in &self.entries {
            out.push_str(&format!(
                "  [{}:{}] offset=0x{:06x} size={}\n",
                e.table_kind, e.entry_index, e.byte_offset, e.size
            ));
        }
        out.push_str("\n=== RIP-relative references ===\n");
        for r in &self.rip_refs {
            out.push_str(&format!(
                "  instr@0x{:06x} disp32={:+} -> target=0x{:06x}\n",
                r.instr_byte_off, r.disp32, r.target_byte_off
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_source_map_new_empty() {
        let map = JitSourceMap::new();

        assert!(map.entries.is_empty());
    }

    #[test]
    fn jit_source_map_default_empty() {
        let map = JitSourceMap::default();

        assert!(map.entries.is_empty());
    }

    #[test]
    fn jit_source_map_add_entry() {
        let mut map = JitSourceMap::new();

        map.add(0x0000_0040, "prefill", "entry_qkv".to_string());

        assert_eq!(map.entries.len(), 1);
        let entry = &map.entries[0];
        assert_eq!(entry.code_offset, 0x40);
        assert_eq!(entry.source.phase, "prefill");
        assert_eq!(entry.source.debug_label.as_deref(), Some("entry_qkv"));
        assert!(entry.source.op_label.is_none());
        assert!(entry.source.fusion_group.is_none());
    }

    #[test]
    fn jit_source_map_add_multiple() {
        let mut map = JitSourceMap::new();

        map.add(0x10, "prefill", "first".to_string());
        map.add(0x20, "decode", "second".to_string());

        assert_eq!(map.entries.len(), 2);
        assert_eq!(map.entries[0].code_offset, 0x10);
        assert_eq!(map.entries[0].source.phase, "prefill");
        assert_eq!(map.entries[1].code_offset, 0x20);
        assert_eq!(map.entries[1].source.phase, "decode");
    }

    #[test]
    fn jit_source_map_sort_by_offset() {
        let mut map = JitSourceMap::new();

        map.add(0x300, "decode", "third".to_string());
        map.add(0x100, "prefill", "first".to_string());
        map.add(0x200, "prefill", "second".to_string());

        map.sort_by_offset();

        assert_eq!(map.entries[0].code_offset, 0x100);
        assert_eq!(map.entries[1].code_offset, 0x200);
        assert_eq!(map.entries[2].code_offset, 0x300);
    }

    #[test]
    fn jit_source_map_to_text_empty() {
        let map = JitSourceMap::new();

        assert!(map.to_text().is_empty());
    }

    #[test]
    fn jit_source_map_to_text_single() {
        let mut map = JitSourceMap::new();
        map.add(0x0000_ab, "prefill", "q_proj".to_string());

        let text = map.to_text();

        assert!(text.contains("0x0000ab"), "hex offset formatted: {text}");
        assert!(text.contains("[     prefill]"), "phase right-aligned in 12-char field: {text}");
        assert!(text.contains("q_proj"), "debug label present: {text}");
        assert!(text.contains("(-)"), "op_label None renders as dash: {text}");
        assert!(text.ends_with('\n'), "line ends with newline: {text}");
    }

    #[test]
    fn jit_source_map_to_text_multiple() {
        let mut map = JitSourceMap::new();
        map.add(0x10, "prefill", "entry_a".to_string());
        map.add(0x20, "decode", "entry_b".to_string());

        let text = map.to_text();

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("0x000010"));
        assert!(lines[0].contains("entry_a"));
        assert!(lines[1].contains("0x000020"));
        assert!(lines[1].contains("entry_b"));
    }

    #[test]
    fn jit_source_map_clone() {
        let mut original = JitSourceMap::new();
        original.add(0x50, "prefill", "cloned_entry".to_string());

        let cloned = original.clone();

        assert_eq!(cloned.entries.len(), 1);
        assert_eq!(cloned.entries[0].code_offset, 0x50);
        assert_eq!(cloned.entries[0].source.phase, "prefill");
        assert_eq!(
            cloned.entries[0].source.debug_label.as_deref(),
            Some("cloned_entry")
        );
    }

    #[test]
    fn source_info_fields() {
        let info = SourceInfo {
            op_label: Some("L0.q_proj".to_string()),
            fusion_group: Some(3),
            phase: "decode".to_string(),
            debug_label: Some("bp_after_rope".to_string()),
        };

        assert_eq!(info.op_label.as_deref(), Some("L0.q_proj"));
        assert_eq!(info.fusion_group, Some(3));
        assert_eq!(info.phase, "decode");
        assert_eq!(info.debug_label.as_deref(), Some("bp_after_rope"));
    }

    #[test]
    fn jit_source_entry_fields() {
        let entry = JitSourceEntry {
            code_offset: 0x1FF,
            source: SourceInfo {
                op_label: Some("embed".to_string()),
                fusion_group: None,
                phase: "prefill".to_string(),
                debug_label: None,
            },
        };

        assert_eq!(entry.code_offset, 0x1FF);
        assert_eq!(entry.source.op_label.as_deref(), Some("embed"));
        assert!(entry.source.fusion_group.is_none());
        assert_eq!(entry.source.phase, "prefill");
        assert!(entry.source.debug_label.is_none());
    }

    // ── Additional tests ───────────────────────────────────────────────────

    #[test]
    fn jit_source_entry_clone() {
        let entry = JitSourceEntry {
            code_offset: 0xAB,
            source: SourceInfo {
                op_label: Some("L2.ffn".to_string()),
                fusion_group: Some(1),
                phase: "decode".to_string(),
                debug_label: Some("bp_after".to_string()),
            },
        };

        let cloned = entry.clone();
        assert_eq!(cloned.code_offset, 0xAB);
        assert_eq!(cloned.source.op_label.as_deref(), Some("L2.ffn"));
        assert_eq!(cloned.source.fusion_group, Some(1));
        assert_eq!(cloned.source.phase, "decode");
        assert_eq!(cloned.source.debug_label.as_deref(), Some("bp_after"));
    }

    #[test]
    fn source_info_clone() {
        let info = SourceInfo {
            op_label: Some("test_op".to_string()),
            fusion_group: Some(5),
            phase: "prefill".to_string(),
            debug_label: Some("marker".to_string()),
        };

        let cloned = info.clone();
        assert_eq!(cloned.op_label, info.op_label);
        assert_eq!(cloned.fusion_group, info.fusion_group);
        assert_eq!(cloned.phase, info.phase);
        assert_eq!(cloned.debug_label, info.debug_label);
    }

    #[test]
    fn jit_source_map_to_text_with_op_label() {
        let mut map = JitSourceMap::new();
        // Manually construct an entry with op_label set
        map.entries.push(JitSourceEntry {
            code_offset: 0x100,
            source: SourceInfo {
                op_label: Some("L0.q_proj".to_string()),
                fusion_group: Some(2),
                phase: "prefill".to_string(),
                debug_label: Some("bp_rope".to_string()),
            },
        });

        let text = map.to_text();
        assert!(text.contains("L0.q_proj"), "op_label should appear in text output");
        assert!(text.contains("bp_rope"), "debug_label should appear");
    }

    #[test]
    fn jit_source_map_sort_preserves_data() {
        let mut map = JitSourceMap::new();
        map.add(0x300, "decode", "third_entry".to_string());
        map.add(0x100, "prefill", "first_entry".to_string());
        map.add(0x200, "prefill", "second_entry".to_string());

        map.sort_by_offset();

        // Verify data integrity after sort
        assert_eq!(map.entries[0].source.debug_label.as_deref(), Some("first_entry"));
        assert_eq!(map.entries[1].source.debug_label.as_deref(), Some("second_entry"));
        assert_eq!(map.entries[2].source.debug_label.as_deref(), Some("third_entry"));
    }

    #[test]
    fn jit_source_map_sort_already_sorted() {
        let mut map = JitSourceMap::new();
        map.add(0x10, "prefill", "a".to_string());
        map.add(0x20, "prefill", "b".to_string());
        map.add(0x30, "decode", "c".to_string());

        map.sort_by_offset();

        // Already sorted order should remain
        assert_eq!(map.entries[0].code_offset, 0x10);
        assert_eq!(map.entries[1].code_offset, 0x20);
        assert_eq!(map.entries[2].code_offset, 0x30);
    }

    #[test]
    fn jit_source_map_sort_empty() {
        let mut map = JitSourceMap::new();
        map.sort_by_offset(); // Should not panic
        assert!(map.entries.is_empty());
    }

    #[test]
    fn jit_source_map_to_text_with_all_none_optional_fields() {
        let mut map = JitSourceMap::new();
        map.entries.push(JitSourceEntry {
            code_offset: 0x50,
            source: SourceInfo {
                op_label: None,
                fusion_group: None,
                phase: "init".to_string(),
                debug_label: None,
            },
        });

        let text = map.to_text();
        // debug_label None => "?", op_label None => "-"
        assert!(text.contains("?"), "None debug_label renders as ?");
        assert!(text.contains("(-)"), "None op_label renders as -");
        assert!(text.contains("[        init]"), "phase 'init' in 12-char field");
    }

    #[test]
    fn jit_source_entry_debug_format() {
        let entry = JitSourceEntry {
            code_offset: 0x42,
            source: SourceInfo {
                op_label: None,
                fusion_group: None,
                phase: "test".to_string(),
                debug_label: None,
            },
        };
        let debug = format!("{:?}", entry);
        assert!(debug.contains("JitSourceEntry"));
        assert!(debug.contains("code_offset"));
    }

    #[test]
    fn source_info_debug_format() {
        let info = SourceInfo {
            op_label: Some("op".to_string()),
            fusion_group: None,
            phase: "p".to_string(),
            debug_label: None,
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("SourceInfo"));
    }

    #[test]
    fn jit_source_map_debug_format() {
        let map = JitSourceMap::new();
        let debug = format!("{:?}", map);
        assert!(debug.contains("JitSourceMap"));
        assert!(debug.contains("entries"));
    }

    // ── VmInstrOffsetMap tests ──────────────────────────────────────────

    #[test]
    fn vm_instr_offset_map_new_empty() {
        let map = VmInstrOffsetMap::new();
        assert!(map.entries.is_empty());
    }

    #[test]
    fn vm_instr_offset_map_default_empty() {
        let map = VmInstrOffsetMap::default();
        assert!(map.entries.is_empty());
    }

    #[test]
    fn vm_instr_offset_map_lookup_hit() {
        let mut map = VmInstrOffsetMap::new();
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 0,
            start_byte_off: 0x0,
            end_byte_off: 0x10,
            instr_debug: "Prologue".to_string(),
        });
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 1,
            start_byte_off: 0x10,
            end_byte_off: 0x20,
            instr_debug: "VecLoad".to_string(),
        });
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 2,
            start_byte_off: 0x20,
            end_byte_off: 0x35,
            instr_debug: "Gemm".to_string(),
        });
        map.sort_by_offset();

        // 查找落在每个区间内的偏移
        assert_eq!(map.lookup_byte_offset(0x0), Some(0));
        assert_eq!(map.lookup_byte_offset(0x0F), Some(0));
        assert_eq!(map.lookup_byte_offset(0x10), Some(1));
        assert_eq!(map.lookup_byte_offset(0x1F), Some(1));
        assert_eq!(map.lookup_byte_offset(0x20), Some(2));
        assert_eq!(map.lookup_byte_offset(0x34), Some(2));
    }

    #[test]
    fn vm_instr_offset_map_lookup_miss() {
        let mut map = VmInstrOffsetMap::new();
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 0,
            start_byte_off: 0x100,
            end_byte_off: 0x200,
            instr_debug: "Instr0".to_string(),
        });

        // 偏移在区间外 (end_byte_off 是半开区间, 0x200 不属于此 entry)
        assert_eq!(map.lookup_byte_offset(0x200), None);
        // 偏移在第一个 entry 之前
        assert_eq!(map.lookup_byte_offset(0x50), None);
    }

    #[test]
    fn vm_instr_offset_map_lookup_empty() {
        let map = VmInstrOffsetMap::new();
        assert_eq!(map.lookup_byte_offset(0x0), None);
    }

    #[test]
    fn vm_instr_offset_map_sort_by_offset() {
        let mut map = VmInstrOffsetMap::new();
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 2,
            start_byte_off: 0x300,
            end_byte_off: 0x400,
            instr_debug: "C".to_string(),
        });
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 0,
            start_byte_off: 0x100,
            end_byte_off: 0x200,
            instr_debug: "A".to_string(),
        });
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 1,
            start_byte_off: 0x200,
            end_byte_off: 0x300,
            instr_debug: "B".to_string(),
        });

        map.sort_by_offset();

        assert_eq!(map.entries[0].vm_instr_index, 0);
        assert_eq!(map.entries[1].vm_instr_index, 1);
        assert_eq!(map.entries[2].vm_instr_index, 2);
    }

    #[test]
    fn vm_instr_offset_map_to_text() {
        let mut map = VmInstrOffsetMap::new();
        map.entries.push(VmInstrOffsetEntry {
            vm_instr_index: 42,
            start_byte_off: 0x1a3f,
            end_byte_off: 0x1a4c,
            instr_debug: "Q5KDecodeStep".to_string(),
        });

        let text = map.to_text();

        assert!(text.contains("VmInstr[  42]"), "vm_instr_index formatted: {text}");
        assert!(text.contains("0x001a3f-0x001a4c"), "byte range formatted: {text}");
        assert!(text.contains("Q5KDecodeStep"), "instr_debug present: {text}");
    }

    #[test]
    fn vm_instr_offset_entry_clone() {
        let entry = VmInstrOffsetEntry {
            vm_instr_index: 5,
            start_byte_off: 0x100,
            end_byte_off: 0x200,
            instr_debug: "Test".to_string(),
        };
        let cloned = entry.clone();
        assert_eq!(cloned.vm_instr_index, 5);
        assert_eq!(cloned.start_byte_off, 0x100);
        assert_eq!(cloned.end_byte_off, 0x200);
        assert_eq!(cloned.instr_debug, "Test");
    }

    // ── ConstPoolAudit tests ────────────────────────────────────────────

    #[test]
    fn const_pool_audit_new_empty() {
        let audit = ConstPoolAudit::new();
        assert!(audit.entries.is_empty());
        assert!(audit.rip_refs.is_empty());
    }

    #[test]
    fn const_pool_audit_default_empty() {
        let audit = ConstPoolAudit::default();
        assert!(audit.entries.is_empty());
        assert!(audit.rip_refs.is_empty());
    }

    #[test]
    fn const_pool_audit_to_text() {
        let mut audit = ConstPoolAudit::new();
        audit.entries.push(PoolTableEntry {
            table_kind: "const_pool".to_string(),
            entry_index: 0,
            byte_offset: 0x1000,
            size: 32,
        });
        audit.rip_refs.push(RipRelRef {
            instr_byte_off: 0x500,
            disp32: 0xAF0,
            target_byte_off: 0x1000,
        });

        let text = audit.to_text();

        assert!(text.contains("ConstPool/DataTables Audit"), "header present: {text}");
        assert!(text.contains("1 entries"), "entry count: {text}");
        assert!(text.contains("1 rip-refs"), "rip-ref count: {text}");
        assert!(text.contains("const_pool:0"), "table kind+index: {text}");
        assert!(text.contains("offset=0x001000"), "entry offset: {text}");
        assert!(text.contains("instr@0x000500"), "rip-ref instr offset: {text}");
        assert!(text.contains("target=0x001000"), "rip-ref target: {text}");
    }

    #[test]
    fn pool_table_entry_clone() {
        let entry = PoolTableEntry {
            table_kind: "data_tables".to_string(),
            entry_index: 3,
            byte_offset: 0x2000,
            size: 64,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.table_kind, "data_tables");
        assert_eq!(cloned.entry_index, 3);
        assert_eq!(cloned.byte_offset, 0x2000);
        assert_eq!(cloned.size, 64);
    }

    #[test]
    fn rip_rel_ref_clone() {
        let r = RipRelRef {
            instr_byte_off: 0x400,
            disp32: -16,
            target_byte_off: 0x3F0,
        };
        let cloned = r.clone();
        assert_eq!(cloned.instr_byte_off, 0x400);
        assert_eq!(cloned.disp32, -16);
        assert_eq!(cloned.target_byte_off, 0x3F0);
    }
}
