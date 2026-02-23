//! Compilation cache — memory + disk persistence for compiled layers.
//!
//! Avoids re-compiling layers when the same model is loaded again on the
//! same hardware. Uses a content-addressed scheme: the cache key is a hash
//! of (LayerIR + DeviceProfile fingerprint).
//!
//! Disk layout (under `~/.cache/gllm-kernels/compiled/`):
//! - `{hash:016x}.bin`  — raw machine code bytes
//! - `{hash:016x}.meta` — JSON metadata: version, scratchpad, config_hash, timestamp

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::compiler::executable::CompiledLayer;
use crate::types::InferenceError;

/// Cache format version — bump when the compiler's codegen changes.
/// Stale entries with a different version are discarded on load.
pub const CACHE_VERSION: &str = "0.1.0";

/// Where a cache hit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    Memory,
    Disk,
}

/// In-memory compilation cache.
///
/// Keyed by config_hash (u64) which encodes the LayerIR + hardware fingerprint.
/// Disk persistence stores raw machine code bytes alongside JSON metadata.
pub struct CompilationCache {
    /// In-memory cache: config_hash → compiled code bytes
    entries: HashMap<u64, CacheEntry>,
    /// Disk cache directory (None = memory-only)
    disk_dir: Option<PathBuf>,
}

struct CacheEntry {
    code_bytes: Vec<u8>,
    scratchpad_bytes: usize,
}

impl CompilationCache {
    /// Create a memory-only cache.
    pub fn new() -> Self {
        CompilationCache {
            entries: HashMap::new(),
            disk_dir: None,
        }
    }

    /// Create a cache with disk persistence.
    pub fn with_disk(dir: PathBuf) -> Self {
        CompilationCache {
            entries: HashMap::new(),
            disk_dir: Some(dir),
        }
    }

    /// Default disk cache location: ~/.cache/gllm-kernels/compiled/
    pub fn default_disk() -> Self {
        let dir = default_cache_dir();
        if let Some(ref d) = dir {
            let _ = std::fs::create_dir_all(d);
        }
        CompilationCache {
            entries: HashMap::new(),
            disk_dir: dir,
        }
    }

    /// Look up a compiled layer by config hash, also reporting the source.
    ///
    /// Returns `None` on miss. On a disk hit the entry is promoted to memory.
    pub fn lookup(&mut self, config_hash: u64) -> Option<(CompiledLayer, CacheSource)> {
        // Hot path: memory
        if let Some(entry) = self.entries.get(&config_hash) {
            let layer = CompiledLayer::from_code(
                &entry.code_bytes,
                entry.scratchpad_bytes,
                config_hash,
            )
            .ok()?;
            return Some((layer, CacheSource::Memory));
        }

        // Cold path: disk
        if let Some(ref dir) = self.disk_dir {
            if let Some(entry) = load_from_disk(dir, config_hash) {
                let layer = CompiledLayer::from_code(
                    &entry.code_bytes,
                    entry.scratchpad_bytes,
                    config_hash,
                )
                .ok();
                // Promote to memory cache
                self.entries.insert(config_hash, entry);
                return layer.map(|l| (l, CacheSource::Disk));
            }
        }

        None
    }

    /// Look up a compiled layer by config hash.
    pub fn get(&mut self, config_hash: u64) -> Option<CompiledLayer> {
        self.lookup(config_hash).map(|(layer, _)| layer)
    }

    /// Store a compiled layer in the cache.
    pub fn put(
        &mut self,
        config_hash: u64,
        code_bytes: &[u8],
        scratchpad_bytes: usize,
    ) {
        let entry = CacheEntry {
            code_bytes: code_bytes.to_vec(),
            scratchpad_bytes,
        };

        // Save to disk
        if let Some(ref dir) = self.disk_dir {
            let _ = save_to_disk(dir, config_hash, &entry);
        }

        self.entries.insert(config_hash, entry);
    }

    /// Number of cached entries (memory).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached entries (memory + disk).
    pub fn clear(&mut self) {
        self.entries.clear();
        if let Some(ref dir) = self.disk_dir {
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::create_dir_all(dir);
        }
    }

    /// Clear only the disk cache, leaving memory entries intact.
    pub fn clear_disk_cache(&mut self) {
        if let Some(ref dir) = self.disk_dir {
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::create_dir_all(dir);
        }
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Disk persistence ────────────────────────────────────────────────

fn cache_file_path(dir: &Path, hash: u64) -> PathBuf {
    dir.join(format!("{hash:016x}.bin"))
}

fn meta_file_path(dir: &Path, hash: u64) -> PathBuf {
    dir.join(format!("{hash:016x}.meta"))
}

/// Disk metadata sidecar (JSON).
struct CacheMeta {
    scratchpad_bytes: usize,
    config_hash: u64,
    version: String,
    timestamp: u64,
}

impl CacheMeta {
    fn to_json(&self) -> String {
        format!(
            "{{\"scratchpad_bytes\":{},\"config_hash\":{},\"version\":\"{}\",\"timestamp\":{}}}",
            self.scratchpad_bytes, self.config_hash, self.version, self.timestamp,
        )
    }

    fn from_json(s: &str) -> Option<Self> {
        let scratchpad_bytes = parse_json_usize(s, "scratchpad_bytes")?;
        let config_hash = parse_json_u64(s, "config_hash")?;
        let version = parse_json_str(s, "version")?;
        let timestamp = parse_json_u64(s, "timestamp")?;
        Some(CacheMeta {
            scratchpad_bytes,
            config_hash,
            version,
            timestamp,
        })
    }
}

fn save_to_disk(dir: &Path, hash: u64, entry: &CacheEntry) -> Result<(), InferenceError> {
    let meta = CacheMeta {
        scratchpad_bytes: entry.scratchpad_bytes,
        config_hash: hash,
        version: CACHE_VERSION.to_string(),
        timestamp: unix_timestamp(),
    };
    // Write code first, then meta (meta presence signals a complete entry)
    std::fs::write(cache_file_path(dir, hash), &entry.code_bytes)?;
    std::fs::write(meta_file_path(dir, hash), meta.to_json())?;
    Ok(())
}

fn load_from_disk(dir: &Path, hash: u64) -> Option<CacheEntry> {
    let code_path = cache_file_path(dir, hash);
    let meta_path = meta_file_path(dir, hash);

    let code_bytes = match std::fs::read(&code_path) {
        Ok(b) if !b.is_empty() => b,
        _ => {
            // No code or empty — clean up orphans
            let _ = std::fs::remove_file(&code_path);
            let _ = std::fs::remove_file(&meta_path);
            return None;
        }
    };

    let meta_str = match std::fs::read_to_string(&meta_path) {
        Ok(s) => s,
        Err(_) => {
            // Orphaned code file
            let _ = std::fs::remove_file(&code_path);
            return None;
        }
    };

    let meta = match CacheMeta::from_json(&meta_str) {
        Some(m) => m,
        None => {
            // Corrupted or old format — discard
            let _ = std::fs::remove_file(&code_path);
            let _ = std::fs::remove_file(&meta_path);
            return None;
        }
    };

    if meta.version != CACHE_VERSION {
        // Stale version — discard
        let _ = std::fs::remove_file(&code_path);
        let _ = std::fs::remove_file(&meta_path);
        return None;
    }

    Some(CacheEntry {
        code_bytes,
        scratchpad_bytes: meta.scratchpad_bytes,
    })
}

fn default_cache_dir() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg).join("gllm-kernels").join("compiled"));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(
            PathBuf::from(home)
                .join(".cache")
                .join("gllm-kernels")
                .join("compiled"),
        );
    }
    None
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Minimal JSON helpers (no serde dependency) ──────────────────────

fn parse_json_usize(json: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| c == ',' || c == '}')?;
    rest[..end].trim().parse().ok()
}

fn parse_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| c == ',' || c == '}')?;
    rest[..end].trim().parse().ok()
}

fn parse_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Compute a config hash from LayerIR fields + hardware fingerprint.
///
/// Uses FNV-1a for speed (not cryptographic — just cache key).
pub fn config_hash(ir_bytes: &[u8], hw_fingerprint: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &b in ir_bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    for &b in hw_fingerprint.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Incremental compilation result ──────────────────────────────────

/// Statistics from an incremental model compilation.
pub struct IncrementalCompileResult {
    /// Compiled layers (one per model layer).
    pub layers: Vec<CompiledLayer>,
    /// Number of layers served from memory cache.
    pub memory_hits: usize,
    /// Number of layers loaded from disk cache.
    pub disk_hits: usize,
    /// Number of layers that required fresh compilation.
    pub compiled: usize,
}

// ── OpTrace persistence ─────────────────────────────────────────────

use crate::compiler::trace::TraceOp;

/// Compact binary representation of a TraceOp.
/// Internal to cache — not exposed publicly.
#[derive(Debug, Clone)]
struct SerializedTraceOp {
    tag: u8,
    operands: [u32; 3],
    float_val: f64,
}

impl SerializedTraceOp {
    fn from_trace_op(op: &TraceOp) -> Self {
        match op {
            TraceOp::Input(n) => Self { tag: 0, operands: [*n, 0, 0], float_val: 0.0 },
            TraceOp::Const(v) => Self { tag: 1, operands: [0, 0, 0], float_val: *v },
            TraceOp::Add(a, b) => Self { tag: 2, operands: [*a, *b, 0], float_val: 0.0 },
            TraceOp::Sub(a, b) => Self { tag: 3, operands: [*a, *b, 0], float_val: 0.0 },
            TraceOp::Mul(a, b) => Self { tag: 4, operands: [*a, *b, 0], float_val: 0.0 },
            TraceOp::Div(a, b) => Self { tag: 5, operands: [*a, *b, 0], float_val: 0.0 },
            TraceOp::Fma(a, b, c) => Self { tag: 6, operands: [*a, *b, *c], float_val: 0.0 },
            TraceOp::Neg(a) => Self { tag: 7, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Abs(a) => Self { tag: 8, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Exp(a) => Self { tag: 9, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Sqrt(a) => Self { tag: 10, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Rsqrt(a) => Self { tag: 11, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Tanh(a) => Self { tag: 12, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Recip(a) => Self { tag: 13, operands: [*a, 0, 0], float_val: 0.0 },
            TraceOp::Max(a, b) => Self { tag: 14, operands: [*a, *b, 0], float_val: 0.0 },
            TraceOp::Min(a, b) => Self { tag: 15, operands: [*a, *b, 0], float_val: 0.0 },
        }
    }

    fn to_trace_op(&self) -> Option<TraceOp> {
        match self.tag {
            0 => Some(TraceOp::Input(self.operands[0])),
            1 => Some(TraceOp::Const(self.float_val)),
            2 => Some(TraceOp::Add(self.operands[0], self.operands[1])),
            3 => Some(TraceOp::Sub(self.operands[0], self.operands[1])),
            4 => Some(TraceOp::Mul(self.operands[0], self.operands[1])),
            5 => Some(TraceOp::Div(self.operands[0], self.operands[1])),
            6 => Some(TraceOp::Fma(self.operands[0], self.operands[1], self.operands[2])),
            7 => Some(TraceOp::Neg(self.operands[0])),
            8 => Some(TraceOp::Abs(self.operands[0])),
            9 => Some(TraceOp::Exp(self.operands[0])),
            10 => Some(TraceOp::Sqrt(self.operands[0])),
            11 => Some(TraceOp::Rsqrt(self.operands[0])),
            12 => Some(TraceOp::Tanh(self.operands[0])),
            13 => Some(TraceOp::Recip(self.operands[0])),
            14 => Some(TraceOp::Max(self.operands[0], self.operands[1])),
            15 => Some(TraceOp::Min(self.operands[0], self.operands[1])),
            _ => None,
        }
    }

    /// Serialize to bytes (fixed 28 bytes per op).
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(28);
        buf.push(self.tag);
        for &op in &self.operands {
            buf.extend_from_slice(&op.to_le_bytes());
        }
        buf.extend_from_slice(&self.float_val.to_le_bytes());
        // Pad to 28 bytes
        while buf.len() < 28 {
            buf.push(0);
        }
        buf
    }

    fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 21 { return None; }
        let tag = data[0];
        let o0 = u32::from_le_bytes(data[1..5].try_into().ok()?);
        let o1 = u32::from_le_bytes(data[5..9].try_into().ok()?);
        let o2 = u32::from_le_bytes(data[9..13].try_into().ok()?);
        let fv = f64::from_le_bytes(data[13..21].try_into().ok()?);
        Some(Self { tag, operands: [o0, o1, o2], float_val: fv })
    }
}

/// Serialize a `Vec<TraceOp>` to bytes (no serde dependency).
pub fn serialize_trace_ops(ops: &[TraceOp]) -> Vec<u8> {
    let mut buf = Vec::new();
    // Header: number of ops (u32)
    buf.extend_from_slice(&(ops.len() as u32).to_le_bytes());
    for op in ops {
        let sop = SerializedTraceOp::from_trace_op(op);
        buf.extend_from_slice(&sop.to_bytes());
    }
    buf
}

/// Deserialize a `Vec<TraceOp>` from bytes.
pub fn deserialize_trace_ops(data: &[u8]) -> Option<Vec<TraceOp>> {
    if data.len() < 4 { return None; }
    let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let mut ops = Vec::with_capacity(count);
    let mut offset = 4;
    for _ in 0..count {
        if offset + 28 > data.len() { return None; }
        let sop = SerializedTraceOp::from_bytes(&data[offset..offset + 28])?;
        ops.push(sop.to_trace_op()?);
        offset += 28;
    }
    Some(ops)
}

/// Compute a hash that includes OpTrace content.
/// This ensures cache invalidation when the scalar op implementation changes.
pub fn trace_aware_hash(ir_bytes: &[u8], hw_fingerprint: &str, trace_ops: &[TraceOp]) -> u64 {
    let mut h = config_hash(ir_bytes, hw_fingerprint);
    // Mix in trace ops
    let trace_bytes = serialize_trace_ops(trace_ops);
    for &b in &trace_bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Cache statistics ────────────────────────────────────────────────

/// Cache statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
    pub total_code_bytes: usize,
    pub num_entries: usize,
}

impl CompilationCache {
    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total_code_bytes: usize = self.entries.values()
            .map(|e| e.code_bytes.len())
            .sum();
        CacheStats {
            memory_hits: 0,
            disk_hits: 0,
            misses: 0,
            total_code_bytes,
            num_entries: self.entries.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_cache() {
        let mut cache = CompilationCache::new();
        assert!(cache.is_empty());

        #[cfg(target_arch = "x86_64")]
        {
            let code = [0xC3u8]; // ret
            cache.put(0x1234, &code, 4096);
            assert_eq!(cache.len(), 1);

            let layer = cache.get(0x1234).unwrap();
            assert_eq!(layer.scratchpad_bytes, 4096);
            assert_eq!(layer.config_hash, 0x1234);
        }
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = CompilationCache::new();
        assert!(cache.get(0xDEAD).is_none());
    }

    #[test]
    fn test_config_hash_deterministic() {
        let h1 = config_hash(b"test_ir_data", "hw_fp_123");
        let h2 = config_hash(b"test_ir_data", "hw_fp_123");
        assert_eq!(h1, h2);

        let h3 = config_hash(b"different_ir", "hw_fp_123");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_lookup_source_memory() {
        let mut cache = CompilationCache::new();

        #[cfg(target_arch = "x86_64")]
        {
            cache.put(0x5555, &[0xC3], 2048);
            let (layer, source) = cache.lookup(0x5555).unwrap();
            assert_eq!(source, CacheSource::Memory);
            assert_eq!(layer.scratchpad_bytes, 2048);
        }
    }

    #[test]
    fn test_disk_cache_roundtrip() {
        let dir = std::env::temp_dir().join("gllm_cache_roundtrip_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let hash: u64 = 0xCAFE_BABE;
        let code = vec![0xC3u8; 64];

        // Write via one cache instance
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(hash, &code, 16384);
        }

        // Verify meta file is proper JSON
        let meta_str = std::fs::read_to_string(meta_file_path(&dir, hash)).unwrap();
        assert!(meta_str.contains("\"scratchpad_bytes\":16384"));
        assert!(meta_str.contains(&format!("\"config_hash\":{}", hash)));
        assert!(meta_str.contains(&format!("\"version\":\"{}\"", CACHE_VERSION)));
        assert!(meta_str.contains("\"timestamp\":"));

        // Fresh cache instance loads from disk
        #[cfg(target_arch = "x86_64")]
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            let (layer, source) = cache.lookup(hash).unwrap();
            assert_eq!(source, CacheSource::Disk);
            assert_eq!(layer.scratchpad_bytes, 16384);
            assert_eq!(layer.config_hash, hash);

            // Second lookup is from memory (promoted)
            let (_, source2) = cache.lookup(hash).unwrap();
            assert_eq!(source2, CacheSource::Memory);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_version_invalidation() {
        let dir = std::env::temp_dir().join("gllm_cache_version_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let hash: u64 = 0x1111_2222;

        // Write files with a stale version
        std::fs::write(cache_file_path(&dir, hash), &[0xC3u8]).unwrap();
        let stale_meta = format!(
            "{{\"scratchpad_bytes\":4096,\"config_hash\":{},\"version\":\"0.0.0-stale\",\"timestamp\":0}}",
            hash,
        );
        std::fs::write(meta_file_path(&dir, hash), stale_meta).unwrap();

        let mut cache = CompilationCache::with_disk(dir.clone());
        assert!(cache.get(hash).is_none());

        // Stale files should have been cleaned up
        assert!(!cache_file_path(&dir, hash).exists());
        assert!(!meta_file_path(&dir, hash).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_corrupted_meta_graceful() {
        let dir = std::env::temp_dir().join("gllm_cache_corrupt_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let hash: u64 = 0x3333_4444;

        std::fs::write(cache_file_path(&dir, hash), &[0xC3u8]).unwrap();
        std::fs::write(meta_file_path(&dir, hash), "not json at all").unwrap();

        let mut cache = CompilationCache::with_disk(dir.clone());
        assert!(cache.get(hash).is_none());

        // Corrupted files cleaned up
        assert!(!cache_file_path(&dir, hash).exists());
        assert!(!meta_file_path(&dir, hash).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_old_format_meta_discarded() {
        let dir = std::env::temp_dir().join("gllm_cache_oldfmt_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let hash: u64 = 0x5555_6666;

        // Old format: meta was just a plain number
        std::fs::write(cache_file_path(&dir, hash), &[0xC3u8]).unwrap();
        std::fs::write(meta_file_path(&dir, hash), "8192").unwrap();

        let mut cache = CompilationCache::with_disk(dir.clone());
        assert!(cache.get(hash).is_none());

        // Old-format files cleaned up
        assert!(!cache_file_path(&dir, hash).exists());
        assert!(!meta_file_path(&dir, hash).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_orphaned_code_no_meta() {
        let dir = std::env::temp_dir().join("gllm_cache_orphan_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let hash: u64 = 0x7777_8888;

        // Code file exists but no meta
        std::fs::write(cache_file_path(&dir, hash), &[0xC3u8]).unwrap();

        let mut cache = CompilationCache::with_disk(dir.clone());
        assert!(cache.get(hash).is_none());

        // Orphaned code file cleaned up
        assert!(!cache_file_path(&dir, hash).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_clear_disk_cache() {
        let dir = std::env::temp_dir().join("gllm_cache_clear_disk_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut cache = CompilationCache::with_disk(dir.clone());
        cache.put(0xAAAA, &[0xC3], 1024);
        cache.put(0xBBBB, &[0xC3], 2048);

        assert_eq!(cache.len(), 2);
        assert!(cache_file_path(&dir, 0xAAAA).exists());
        assert!(cache_file_path(&dir, 0xBBBB).exists());

        cache.clear_disk_cache();

        // Memory cache still intact
        assert_eq!(cache.len(), 2);
        // Disk files gone
        assert!(!cache_file_path(&dir, 0xAAAA).exists());
        assert!(!cache_file_path(&dir, 0xBBBB).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_meta_json_roundtrip() {
        let meta = CacheMeta {
            scratchpad_bytes: 65536,
            config_hash: 0xDEAD_BEEF,
            version: "0.1.0".to_string(),
            timestamp: 1700000000,
        };
        let json = meta.to_json();
        let parsed = CacheMeta::from_json(&json).unwrap();
        assert_eq!(parsed.scratchpad_bytes, 65536);
        assert_eq!(parsed.config_hash, 0xDEAD_BEEF);
        assert_eq!(parsed.version, "0.1.0");
        assert_eq!(parsed.timestamp, 1700000000);
    }

    #[test]
    fn test_disk_cache_legacy_compat() {
        // The old test_disk_cache still works
        let dir = std::env::temp_dir().join("gllm_compile_cache_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(0xABCD, &[0xC3], 8192);
        }

        #[cfg(target_arch = "x86_64")]
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            let layer = cache.get(0xABCD).unwrap();
            assert_eq!(layer.scratchpad_bytes, 8192);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── OpTrace persistence tests ───────────────────────────────────

    #[test]
    fn test_serialize_trace_ops_roundtrip() {
        use crate::compiler::trace::TraceOp;

        let ops = vec![
            TraceOp::Input(0),
            TraceOp::Neg(0),
            TraceOp::Exp(1),
            TraceOp::Const(1.0),
            TraceOp::Add(2, 3),
            TraceOp::Div(0, 4),
            TraceOp::Fma(0, 1, 2),
            TraceOp::Max(3, 4),
            TraceOp::Min(0, 5),
            TraceOp::Rsqrt(2),
            TraceOp::Tanh(1),
            TraceOp::Recip(0),
            TraceOp::Abs(3),
            TraceOp::Sqrt(2),
            TraceOp::Sub(0, 1),
            TraceOp::Mul(2, 3),
        ];

        let bytes = serialize_trace_ops(&ops);
        let decoded = deserialize_trace_ops(&bytes).unwrap();

        assert_eq!(ops.len(), decoded.len());
        for (orig, dec) in ops.iter().zip(decoded.iter()) {
            assert_eq!(orig, dec, "mismatch: {orig:?} vs {dec:?}");
        }
    }

    #[test]
    fn test_serialize_empty() {
        let bytes = serialize_trace_ops(&[]);
        let decoded = deserialize_trace_ops(&bytes).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_trace_aware_hash_differs() {
        use crate::compiler::trace::TraceOp;

        let ops1 = vec![TraceOp::Input(0), TraceOp::Neg(0)];
        let ops2 = vec![TraceOp::Input(0), TraceOp::Exp(0)];

        let h1 = trace_aware_hash(b"test", "hw", &ops1);
        let h2 = trace_aware_hash(b"test", "hw", &ops2);

        assert_ne!(h1, h2, "Different traces should produce different hashes");
    }

    #[test]
    fn test_trace_aware_hash_deterministic() {
        use crate::compiler::trace::TraceOp;

        let ops = vec![TraceOp::Input(0), TraceOp::Const(3.14), TraceOp::Mul(0, 1)];

        let h1 = trace_aware_hash(b"ir", "hw_fp", &ops);
        let h2 = trace_aware_hash(b"ir", "hw_fp", &ops);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = CompilationCache::new();
        let stats = cache.stats();
        assert_eq!(stats.num_entries, 0);
        assert_eq!(stats.total_code_bytes, 0);

        cache.put(0x1111, &[0xC3; 100], 4096);
        cache.put(0x2222, &[0xC3; 200], 8192);

        let stats = cache.stats();
        assert_eq!(stats.num_entries, 2);
        assert_eq!(stats.total_code_bytes, 300);
    }

    #[test]
    fn test_serialized_trace_op_all_variants() {
        use crate::compiler::trace::TraceOp;

        let all_ops = vec![
            TraceOp::Input(42),
            TraceOp::Const(std::f64::consts::PI),
            TraceOp::Add(0, 1),
            TraceOp::Sub(2, 3),
            TraceOp::Mul(4, 5),
            TraceOp::Div(6, 7),
            TraceOp::Fma(0, 1, 2),
            TraceOp::Neg(3),
            TraceOp::Abs(4),
            TraceOp::Exp(5),
            TraceOp::Sqrt(6),
            TraceOp::Rsqrt(7),
            TraceOp::Tanh(8),
            TraceOp::Recip(9),
            TraceOp::Max(0, 1),
            TraceOp::Min(2, 3),
        ];

        // Each op should roundtrip correctly
        for op in &all_ops {
            let sop = SerializedTraceOp::from_trace_op(op);
            let bytes = sop.to_bytes();
            let decoded_sop = SerializedTraceOp::from_bytes(&bytes).unwrap();
            let decoded_op = decoded_sop.to_trace_op().unwrap();
            assert_eq!(op, &decoded_op, "roundtrip failed for {op:?}");
        }
    }
}
