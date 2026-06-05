/// JIT Source Map — VmInstr → 机器码偏移 → Op 标签映射。
/// 仅当 MegaKernelBusinessConfig.debug_jit=true 时生成。
/// DAP 调试器用它做 source → address 映射。

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
}
