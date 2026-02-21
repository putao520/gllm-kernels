//! Compilation cache — memory + disk persistence for compiled layers.
//!
//! Avoids re-compiling layers when the same model is loaded again on the
//! same hardware. Uses a content-addressed scheme: the cache key is a hash
//! of (LayerIR + DeviceProfile fingerprint).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::compiler::executable::CompiledLayer;
use crate::inference::types::InferenceError;

/// In-memory compilation cache.
///
/// Keyed by config_hash (u64) which encodes the LayerIR + hardware fingerprint.
/// Disk persistence stores raw machine code bytes alongside metadata.
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

    /// Look up a compiled layer by config hash.
    pub fn get(&mut self, config_hash: u64) -> Option<CompiledLayer> {
        // Check memory first
        if let Some(entry) = self.entries.get(&config_hash) {
            return CompiledLayer::from_code(
                &entry.code_bytes,
                entry.scratchpad_bytes,
                config_hash,
            )
            .ok();
        }

        // Check disk
        if let Some(ref dir) = self.disk_dir {
            if let Some(entry) = load_from_disk(dir, config_hash) {
                let result = CompiledLayer::from_code(
                    &entry.code_bytes,
                    entry.scratchpad_bytes,
                    config_hash,
                )
                .ok();
                // Promote to memory cache
                self.entries.insert(config_hash, entry);
                return result;
            }
        }

        None
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

fn save_to_disk(dir: &Path, hash: u64, entry: &CacheEntry) -> Result<(), InferenceError> {
    std::fs::write(cache_file_path(dir, hash), &entry.code_bytes)?;
    let meta = format!("{}", entry.scratchpad_bytes);
    std::fs::write(meta_file_path(dir, hash), meta)?;
    Ok(())
}

fn load_from_disk(dir: &Path, hash: u64) -> Option<CacheEntry> {
    let code_bytes = std::fs::read(cache_file_path(dir, hash)).ok()?;
    let meta_str = std::fs::read_to_string(meta_file_path(dir, hash)).ok()?;
    let scratchpad_bytes: usize = meta_str.trim().parse().ok()?;
    Some(CacheEntry {
        code_bytes,
        scratchpad_bytes,
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
    fn test_disk_cache() {
        let dir = std::env::temp_dir().join("gllm_compile_cache_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(0xABCD, &[0xC3], 8192);
        }

        // New cache instance should find it on disk
        #[cfg(target_arch = "x86_64")]
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            let layer = cache.get(0xABCD).unwrap();
            assert_eq!(layer.scratchpad_bytes, 8192);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
