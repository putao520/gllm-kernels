//! Parameter cache — FFTW wisdom-style persistence.
//!
//! Tuned parameters are saved to a JSON file keyed by hardware fingerprint
//! and problem shape. On subsequent runs, cached parameters are loaded
//! instantly instead of re-tuning.
//!
//! JIT-specific parameters (k_unroll, prefetch, reg_alloc, sw_pipeline, nr_variant)
//! are serialized alongside the base blocking parameters when present.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::autotuning::search_space::{ProblemShape, TuningConfig, JitParams, RegAllocStrategy};
use crate::types::CompilerError;

/// A single cached tuning result.
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub config: TuningConfig,
    pub median_ns: f64,
    pub gflops: Option<f64>,
    /// Timestamp when this result was recorded (unix seconds)
    pub timestamp: u64,
}

/// The full wisdom database: hw_fingerprint -> (op_key -> CachedResult).
#[derive(Debug, Clone)]
pub struct WisdomDb {
    /// Map: hw_fingerprint -> (op_key -> CachedResult)
    entries: HashMap<String, HashMap<String, CachedResult>>,
    /// File path for persistence
    path: PathBuf,
    /// Whether the DB has been modified since last save
    dirty: bool,
}

impl WisdomDb {
    /// Create a new empty wisdom database at the given path.
    pub fn new(path: PathBuf) -> Self {
        WisdomDb {
            entries: HashMap::new(),
            path,
            dirty: false,
        }
    }

    /// Default wisdom file location: ~/.cache/gllm-kernels/wisdom.json
    pub fn default_path() -> PathBuf {
        if let Some(cache_dir) = dirs_cache() {
            cache_dir.join("gllm-kernels").join("wisdom.json")
        } else {
            PathBuf::from("gllm_wisdom.json")
        }
    }

    /// Load wisdom from disk. Returns empty DB if file doesn't exist.
    pub fn load(path: &Path) -> Self {
        let mut db = WisdomDb::new(path.to_path_buf());
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(parsed) = parse_wisdom(&content) {
                db.entries = parsed;
            }
        }
        db
    }

    /// Load from default location.
    pub fn load_default() -> Self {
        Self::load(&Self::default_path())
    }

    /// Look up a cached result.
    pub fn get(&self, hw_fingerprint: &str, op_key: &str) -> Option<&CachedResult> {
        self.entries
            .get(hw_fingerprint)
            .and_then(|m| m.get(op_key))
    }

    /// Insert or update a cached result.
    pub fn put(
        &mut self,
        hw_fingerprint: &str,
        op_key: &str,
        config: TuningConfig,
        median_ns: f64,
        gflops: Option<f64>,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let hw_map = self
            .entries
            .entry(hw_fingerprint.to_string())
            .or_default();
        hw_map.insert(
            op_key.to_string(),
            CachedResult {
                config,
                median_ns,
                gflops,
                timestamp,
            },
        );
        self.dirty = true;
    }

    /// Save wisdom to disk (only if modified).
    pub fn save(&mut self) -> std::io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serialize_wisdom(&self.entries);
        std::fs::write(&self.path, content)?;
        self.dirty = false;
        Ok(())
    }

    /// Number of total entries across all hardware fingerprints.
    pub fn total_entries(&self) -> usize {
        self.entries.values().map(|m| m.len()).sum()
    }

    /// List all hardware fingerprints in the database.
    pub fn fingerprints(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all entries for a specific hardware fingerprint.
    pub fn clear_hw(&mut self, hw_fingerprint: &str) {
        if self.entries.remove(hw_fingerprint).is_some() {
            self.dirty = true;
        }
    }

    /// Clear the entire database.
    pub fn clear_all(&mut self) {
        if !self.entries.is_empty() {
            self.entries.clear();
            self.dirty = true;
        }
    }

    /// Look up a cached GEMM blocking result and return it as a GemmBlocking.
    ///
    /// Used by `DeviceProfile::gemm_blocking()` to prefer empirically-tuned
    /// parameters over heuristic defaults.
    ///
    /// The `dtype_id` parameter isolates entries by data type (F16 vs BF16
    /// share elem_bytes=2 but have different optimal blocking). Must match
    /// the `elem_id()` of the `DType` used during tuning.
    pub fn get_gemm_blocking(
        &self,
        hw_fingerprint: &str,
        m: usize,
        n: usize,
        k: usize,
        elem_bytes: usize,
        dtype_id: u8,
    ) -> Option<&CachedResult> {
        // Try jit_gemm key first (includes dtype_id from tune_jit_gemm)
        let jit_shape = ProblemShape { m, n, k, elem_bytes, dtype_id };
        let jit_key = op_key("jit_gemm", &jit_shape);
        if let Some(result) = self.get(hw_fingerprint, &jit_key) {
            return Some(result);
        }
        // Fallback: legacy tune_gemm entries (always dtype_id=0)
        // Only match if the query dtype matches F32 (elem_id=0)
        if dtype_id == 0 {
            let std_shape = ProblemShape { m, n, k, elem_bytes, dtype_id: 0 };
            let std_key = op_key("gemm", &std_shape);
            if let Some(result) = self.get(hw_fingerprint, &std_key) {
                return Some(result);
            }
        }
        None
    }
}

/// Generate the operation key for cache lookup.
pub fn op_key(op_name: &str, shape: &ProblemShape) -> String {
    format!("{op_name}_{shape}")
}

// ── Serialization (simple JSON, no serde dependency) ────────────────────

fn serialize_wisdom(
    entries: &HashMap<String, HashMap<String, CachedResult>>,
) -> String {
    let mut out = String::from("{\n");
    let hw_keys: Vec<&String> = {
        let mut v: Vec<_> = entries.keys().collect();
        v.sort();
        v
    };
    for (hi, hw_key) in hw_keys.iter().enumerate() {
        let hw_map = &entries[*hw_key];
        out.push_str(&format!("  \"{}\": {{\n", escape_json(hw_key)));
        let op_keys: Vec<&String> = {
            let mut v: Vec<_> = hw_map.keys().collect();
            v.sort();
            v
        };
        for (oi, op_key) in op_keys.iter().enumerate() {
            let r = &hw_map[*op_key];
            out.push_str(&format!(
                "    \"{}\": {{\"kc\":{},\"mc\":{},\"nc\":{},\"threads\":{},\"median_ns\":{:.1},\"gflops\":{},\"ts\":{}",
                escape_json(op_key),
                r.config.kc,
                r.config.mc,
                r.config.nc,
                r.config.num_threads,
                r.median_ns,
                r.gflops.map(|g| format!("{g:.2}")).unwrap_or_else(|| "null".into()),
                r.timestamp,
            ));
            // Serialize JIT params if present
            if let Some(jit) = &r.config.jit {
                out.push_str(&format!(
                    ",\"k_unroll\":{},\"prefetch\":{},\"reg_alloc\":{},\"sw_pipeline\":{},\"nr_variant\":{}",
                    jit.k_unroll,
                    jit.prefetch_distance,
                    jit.reg_alloc_strategy.to_index(),
                    jit.sw_pipeline_depth,
                    jit.nr_variant,
                ));
            }
            out.push('}');
            if oi + 1 < op_keys.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  }");
        if hi + 1 < hw_keys.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push('}');
    out
}

fn parse_wisdom(
    content: &str,
) -> Result<HashMap<String, HashMap<String, CachedResult>>, CompilerError> {
    // Minimal JSON parser for our known schema.
    let mut result: HashMap<String, HashMap<String, CachedResult>> = HashMap::new();

    let content = content.trim();
    if !content.starts_with('{') || !content.ends_with('}') {
        return Err("Invalid JSON: not an object".into());
    }
    let inner = &content[1..content.len() - 1];

    // Split by top-level hw fingerprint entries
    let hw_entries = split_json_object(inner);
    for (hw_key, hw_value) in hw_entries {
        let hw_value = hw_value.trim();
        if !hw_value.starts_with('{') || !hw_value.ends_with('}') {
            continue;
        }
        let hw_inner = &hw_value[1..hw_value.len() - 1];
        let op_entries = split_json_object(hw_inner);
        let mut hw_map = HashMap::new();
        for (op_key, op_value) in op_entries {
            if let Some(cached) = parse_cached_result(&op_value) {
                hw_map.insert(op_key, cached);
            }
        }
        result.insert(hw_key, hw_map);
    }

    Ok(result)
}

fn parse_cached_result(json: &str) -> Option<CachedResult> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return None;
    }
    let inner = &json[1..json.len() - 1];

    let kc = extract_usize(inner, "kc")?;
    let mc = extract_usize(inner, "mc")?;
    let nc = extract_usize(inner, "nc")?;
    let threads = extract_usize(inner, "threads")?;
    let median_ns = extract_f64(inner, "median_ns")?;
    let gflops = extract_f64(inner, "gflops"); // optional
    let timestamp = extract_usize(inner, "ts").unwrap_or(0) as u64;

    // Parse optional JIT params
    let jit = parse_jit_params(inner);

    Some(CachedResult {
        config: TuningConfig {
            kc,
            mc,
            nc,
            num_threads: threads,
            jit,
        },
        median_ns,
        gflops,
        timestamp,
    })
}

/// Parse JIT parameters from a JSON object interior.
/// Returns None if any JIT field is missing (backward compatible with old entries).
fn parse_jit_params(inner: &str) -> Option<JitParams> {
    let k_unroll = extract_usize(inner, "k_unroll")?;
    let prefetch_distance = extract_usize(inner, "prefetch")?;
    let reg_alloc_idx = extract_usize(inner, "reg_alloc")?;
    let sw_pipeline_depth = extract_usize(inner, "sw_pipeline")?;
    let nr_variant = extract_usize(inner, "nr_variant")?;

    Some(JitParams {
        k_unroll,
        prefetch_distance,
        reg_alloc_strategy: RegAllocStrategy::from_index(reg_alloc_idx),
        sw_pipeline_depth,
        nr_variant,
    })
}

fn extract_usize(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
    after[..end].parse().ok()
}

fn extract_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    if after.starts_with("null") {
        return None;
    }
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+')
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

fn split_json_object(s: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    let mut depth = 0i32;
    let _in_string = false;
    let mut escape = false;
    let mut current_key = String::new();
    let mut current_value = String::new();
    let mut reading_key = false;
    let mut reading_value = false;
    let mut key_done = false;

    for ch in s.chars() {
        if escape {
            if reading_key {
                current_key.push(ch);
            } else if reading_value {
                current_value.push(ch);
            }
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            if reading_key {
                current_key.push(ch);
            } else if reading_value {
                current_value.push(ch);
            }
            continue;
        }

        if ch == '"' && depth == 0 && !key_done {
            if !reading_key {
                reading_key = true;
                current_key.clear();
            } else {
                reading_key = false;
                key_done = true;
            }
            continue;
        }

        if reading_key {
            current_key.push(ch);
            continue;
        }

        if key_done && ch == ':' && depth == 0 {
            reading_value = true;
            current_value.clear();
            continue;
        }

        if reading_value {
            if ch == '{' || ch == '[' {
                depth += 1;
            }
            if ch == '}' || ch == ']' {
                depth -= 1;
            }
            if ch == ',' && depth == 0 {
                result.push((current_key.clone(), current_value.trim().to_string()));
                current_key.clear();
                current_value.clear();
                reading_value = false;
                key_done = false;
                continue;
            }
            current_value.push(ch);
        }
    }

    if key_done && !current_value.trim().is_empty() {
        result.push((current_key, current_value.trim().to_string()));
    }

    result
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn dirs_cache() -> Option<PathBuf> {
    // XDG_CACHE_HOME or ~/.cache
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(PathBuf::from(home).join(".cache"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let dir = std::env::temp_dir().join("gllm_wisdom_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_wisdom.json");

        let mut db = WisdomDb::new(path.clone());
        let config = TuningConfig {
            kc: 256,
            mc: 72,
            nc: 1024,
            num_threads: 8,
            jit: None,
        };
        db.put("test_hw_fp", "gemm_512x512x512_e4", config.clone(), 1234.5, Some(42.0));
        db.save().unwrap();

        // Reload
        let db2 = WisdomDb::load(&path);
        let cached = db2.get("test_hw_fp", "gemm_512x512x512_e4").unwrap();
        assert_eq!(cached.config.kc, 256);
        assert_eq!(cached.config.mc, 72);
        assert_eq!(cached.config.nc, 1024);
        assert_eq!(cached.config.num_threads, 8);
        assert!(cached.config.jit.is_none());
        assert!((cached.median_ns - 1234.5).abs() < 0.1);
        assert!((cached.gflops.unwrap() - 42.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_roundtrip_jit_params() {
        let dir = std::env::temp_dir().join("gllm_wisdom_test_jit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_wisdom_jit.json");

        let mut db = WisdomDb::new(path.clone());
        let config = TuningConfig {
            kc: 128,
            mc: 48,
            nc: 512,
            num_threads: 4,
            jit: Some(JitParams {
                k_unroll: 4,
                prefetch_distance: 8,
                reg_alloc_strategy: RegAllocStrategy::Balanced,
                sw_pipeline_depth: 1,
                nr_variant: 16,
            }),
        };
        db.put("test_hw_jit", "jit_gemm_256x256x256_e4", config.clone(), 500.0, Some(80.0));
        db.save().unwrap();

        let db2 = WisdomDb::load(&path);
        let cached = db2.get("test_hw_jit", "jit_gemm_256x256x256_e4").unwrap();
        assert_eq!(cached.config.kc, 128);
        let jit = cached.config.jit.as_ref().unwrap();
        assert_eq!(jit.k_unroll, 4);
        assert_eq!(jit.prefetch_distance, 8);
        assert_eq!(jit.reg_alloc_strategy, RegAllocStrategy::Balanced);
        assert_eq!(jit.sw_pipeline_depth, 1);
        assert_eq!(jit.nr_variant, 16);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multiple_entries() {
        let dir = std::env::temp_dir().join("gllm_wisdom_test2");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_wisdom2.json");

        let mut db = WisdomDb::new(path.clone());
        let c1 = TuningConfig { kc: 128, mc: 48, nc: 512, num_threads: 4, jit: None };
        let c2 = TuningConfig { kc: 256, mc: 96, nc: 2048, num_threads: 8, jit: None };
        db.put("hw1", "gemm_256x256x256_e4", c1, 500.0, Some(10.0));
        db.put("hw1", "gemm_1024x1024x1024_e4", c2, 2000.0, Some(80.0));
        db.put("hw2", "gemm_256x256x256_e4", TuningConfig { kc: 64, mc: 24, nc: 256, num_threads: 2, jit: None }, 800.0, None);
        db.save().unwrap();

        let db2 = WisdomDb::load(&path);
        assert_eq!(db2.total_entries(), 3);
        assert_eq!(db2.fingerprints().len(), 2);
        assert!(db2.get("hw1", "gemm_256x256x256_e4").is_some());
        assert!(db2.get("hw2", "gemm_256x256x256_e4").is_some());
        assert!(db2.get("hw1", "gemm_1024x1024x1024_e4").is_some());
        assert!(db2.get("hw2", "nonexistent").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_op_key_format() {
        let shape = ProblemShape { m: 512, n: 1024, k: 768, elem_bytes: 4, dtype_id: 0 };
        let key = op_key("gemm", &shape);
        assert_eq!(key, "gemm_512x1024x768_e4_d0");
    }

    #[test]
    fn test_get_gemm_blocking() {
        let dir = std::env::temp_dir().join("gllm_wisdom_blocking");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_blocking.json");

        let mut db = WisdomDb::new(path);
        let config = TuningConfig { kc: 256, mc: 72, nc: 1024, num_threads: 8, jit: None };
        db.put("hw_test", "gemm_512x512x512_e4_d0", config, 1000.0, Some(50.0));

        let result = db.get_gemm_blocking("hw_test", 512, 512, 512, 4, 0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().config.kc, 256);

        // Non-existent shape returns None
        assert!(db.get_gemm_blocking("hw_test", 1024, 1024, 1024, 4, 0).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 13 new tests (+5 existing = 18 total) ──────────────────────────────

    #[test]
    fn test_cached_result_constructor_fields() {
        // Arrange: build a CachedResult with known values
        let config = TuningConfig {
            kc: 64,
            mc: 32,
            nc: 256,
            num_threads: 2,
            jit: None,
        };
        let ts = 1_700_000_000u64;

        // Act: construct directly via struct literal
        let result = CachedResult {
            config: config.clone(),
            median_ns: 99.9,
            gflops: None,
            timestamp: ts,
        };

        // Assert: every field round-trips exactly
        assert_eq!(result.config, config);
        assert!((result.median_ns - 99.9).abs() < f64::EPSILON);
        assert!(result.gflops.is_none());
        assert_eq!(result.timestamp, ts);
    }

    #[test]
    fn test_cached_result_debug_clone_derives() {
        // Arrange
        let config = TuningConfig {
            kc: 128,
            mc: 64,
            nc: 512,
            num_threads: 4,
            jit: Some(JitParams::default()),
        };
        let original = CachedResult {
            config,
            median_ns: 500.0,
            gflops: Some(33.3),
            timestamp: 12345,
        };

        // Act: Clone
        let cloned = original.clone();

        // Assert: cloned equals original
        assert_eq!(cloned.config.kc, original.config.kc);
        assert_eq!(cloned.config.mc, original.config.mc);
        assert_eq!(cloned.config.nc, original.config.nc);
        assert_eq!(cloned.config.num_threads, original.config.num_threads);
        assert!(cloned.config.jit.is_some());
        assert!((cloned.median_ns - original.median_ns).abs() < f64::EPSILON);
        assert_eq!(cloned.gflops, original.gflops);
        assert_eq!(cloned.timestamp, original.timestamp);

        // Act: Debug formatting does not panic
        let debug_str = format!("{:?}", original);
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("CachedResult"));
    }

    #[test]
    fn test_wisdom_db_new_is_clean() {
        // Arrange & Act
        let db = WisdomDb::new(PathBuf::from("/tmp/never_used.json"));

        // Assert: empty, not dirty, correct path
        assert_eq!(db.total_entries(), 0);
        assert!(db.fingerprints().is_empty());
        assert!(db.get("any", "any").is_none());
        assert_eq!(db.path, PathBuf::from("/tmp/never_used.json"));
        // dirty is private; indirectly assert by calling save (no-op if clean)
        let mut db_mut = db;
        db_mut.save().unwrap(); // no-op, does not create file
        assert!(!Path::new("/tmp/never_used.json").exists());
    }

    #[test]
    fn test_wisdom_db_put_overwrite_same_key() {
        // Arrange
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let c1 = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };
        let c2 = TuningConfig { kc: 128, mc: 64, nc: 512, num_threads: 4, jit: None };

        // Act: insert then overwrite the same key
        db.put("hw", "op_a", c1, 100.0, Some(10.0));
        db.put("hw", "op_a", c2.clone(), 200.0, Some(20.0));

        // Assert: only one entry, values are from the second put
        assert_eq!(db.total_entries(), 1);
        let cached = db.get("hw", "op_a").unwrap();
        assert_eq!(cached.config.kc, 128);
        assert_eq!(cached.config.mc, 64);
        assert!((cached.median_ns - 200.0).abs() < f64::EPSILON);
        assert!((cached.gflops.unwrap() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wisdom_db_clear_hw_selective() {
        // Arrange
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };
        db.put("hw_a", "op1", cfg.clone(), 100.0, None);
        db.put("hw_a", "op2", cfg.clone(), 200.0, None);
        db.put("hw_b", "op1", cfg.clone(), 300.0, None);
        assert_eq!(db.total_entries(), 3);

        // Act: clear only hw_a
        db.clear_hw("hw_a");

        // Assert: hw_a gone, hw_b intact
        assert_eq!(db.total_entries(), 1);
        assert!(db.get("hw_a", "op1").is_none());
        assert!(db.get("hw_b", "op1").is_some());

        // Act: clearing non-existent fingerprint is a no-op
        db.clear_hw("nonexistent");
        assert_eq!(db.total_entries(), 1);
    }

    #[test]
    fn test_wisdom_db_clear_all() {
        // Arrange
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };
        db.put("hw1", "op1", cfg.clone(), 100.0, None);
        db.put("hw2", "op2", cfg.clone(), 200.0, None);
        assert_eq!(db.total_entries(), 2);

        // Act
        db.clear_all();

        // Assert
        assert_eq!(db.total_entries(), 0);
        assert!(db.fingerprints().is_empty());

        // Clearing empty DB is a no-op (does not set dirty)
        db.clear_all();
        assert_eq!(db.total_entries(), 0);
    }

    #[test]
    fn test_wisdom_db_fingerprints_ordering() {
        // Arrange
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };

        // Act: insert in non-alphabetical order
        db.put("charlie", "op1", cfg.clone(), 100.0, None);
        db.put("alpha", "op2", cfg.clone(), 200.0, None);
        db.put("bravo", "op3", cfg.clone(), 300.0, None);

        // Assert: fingerprints returns all keys (HashMap order is not guaranteed,
        // but all keys must be present)
        let fps = db.fingerprints();
        assert_eq!(fps.len(), 3);
        let mut sorted: Vec<&str> = fps;
        sorted.sort();
        assert_eq!(sorted, vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn test_cached_result_float_precision() {
        // Arrange: use values that are tricky for floating-point
        let config = TuningConfig { kc: 1, mc: 1, nc: 1, num_threads: 1, jit: None };
        let median = 0.123456789012345f64;
        let gflops = 1e15f64;

        // Act
        let result = CachedResult {
            config,
            median_ns: median,
            gflops: Some(gflops),
            timestamp: 0,
        };

        // Assert: exact bit-level equality for f64
        assert_eq!(result.median_ns.to_bits(), median.to_bits());
        assert_eq!(result.gflops.unwrap().to_bits(), gflops.to_bits());
    }

    #[test]
    fn test_cached_result_timestamp_zero_boundary() {
        // Arrange: timestamp = 0 (UNIX epoch boundary)
        let config = TuningConfig { kc: 1, mc: 1, nc: 1, num_threads: 1, jit: None };
        let result = CachedResult {
            config,
            median_ns: 0.0,
            gflops: Some(0.0),
            timestamp: 0,
        };

        // Assert: zero values are valid, not confused with "missing"
        assert_eq!(result.timestamp, 0);
        assert_eq!(result.median_ns, 0.0);
        assert!(result.gflops.is_some());
        assert_eq!(result.gflops.unwrap(), 0.0);
    }

    #[test]
    fn test_cached_result_usize_max_values() {
        // Arrange: use usize::MAX for blocking params to verify no overflow
        let config = TuningConfig {
            kc: usize::MAX,
            mc: usize::MAX,
            nc: usize::MAX,
            num_threads: usize::MAX,
            jit: None,
        };
        let result = CachedResult {
            config,
            median_ns: f64::MAX,
            gflops: Some(f64::MIN_POSITIVE),
            timestamp: u64::MAX,
        };

        // Assert: extreme values survive struct construction
        assert_eq!(result.config.kc, usize::MAX);
        assert_eq!(result.config.mc, usize::MAX);
        assert_eq!(result.config.nc, usize::MAX);
        assert_eq!(result.config.num_threads, usize::MAX);
        assert_eq!(result.timestamp, u64::MAX);
        assert_eq!(result.median_ns, f64::MAX);
        assert_eq!(result.gflops.unwrap(), f64::MIN_POSITIVE);
    }

    #[test]
    fn test_cached_result_struct_update_syntax_with_jit() {
        // Arrange: base config with JIT params
        let jit = JitParams {
            k_unroll: 2,
            prefetch_distance: 4,
            reg_alloc_strategy: RegAllocStrategy::MinSpill,
            sw_pipeline_depth: 2,
            nr_variant: 8,
        };
        let base = CachedResult {
            config: TuningConfig {
                kc: 256,
                mc: 96,
                nc: 1024,
                num_threads: 8,
                jit: Some(jit.clone()),
            },
            median_ns: 5000.0,
            gflops: Some(100.0),
            timestamp: 9999,
        };

        // Act: use struct update syntax to override selected fields
        let updated = CachedResult {
            median_ns: 3000.0,
            gflops: None,
            ..base
        };

        // Assert: overridden fields changed, rest inherited
        assert!((updated.median_ns - 3000.0).abs() < f64::EPSILON);
        assert!(updated.gflops.is_none());
        assert_eq!(updated.config.kc, 256);
        assert_eq!(updated.config.jit.as_ref().unwrap().k_unroll, 2);
        assert_eq!(updated.config.jit.as_ref().unwrap().reg_alloc_strategy, RegAllocStrategy::MinSpill);
        assert_eq!(updated.timestamp, 9999);
    }

    #[test]
    fn test_reg_alloc_strategy_roundtrip_all_variants() {
        // Arrange: all three variants
        let variants = RegAllocStrategy::all();

        // Act & Assert: each variant survives to_index -> from_index roundtrip
        for &variant in variants {
            let idx = variant.to_index();
            let restored = RegAllocStrategy::from_index(idx);
            assert_eq!(variant, restored, "roundtrip failed for {:?}", variant);
        }

        // Assert: scratch_regs values are distinct and reasonable
        assert_eq!(RegAllocStrategy::MaxAccumulators.scratch_regs(), 2);
        assert_eq!(RegAllocStrategy::Balanced.scratch_regs(), 4);
        assert_eq!(RegAllocStrategy::MinSpill.scratch_regs(), 6);
    }

    #[test]
    fn test_jit_params_default_values() {
        // Arrange & Act
        let default = JitParams::default();

        // Assert: matches the documented default
        assert_eq!(default.k_unroll, 4);
        assert_eq!(default.prefetch_distance, 8);
        assert_eq!(default.reg_alloc_strategy, RegAllocStrategy::Balanced);
        assert_eq!(default.sw_pipeline_depth, 0);
        assert_eq!(default.nr_variant, 16);
    }

    #[test]
    fn test_problem_shape_equality_and_format() {
        // Arrange
        let a = ProblemShape { m: 128, n: 256, k: 512, elem_bytes: 2, dtype_id: 1 };
        let b = ProblemShape { m: 128, n: 256, k: 512, elem_bytes: 2, dtype_id: 1 };
        let c = ProblemShape { m: 128, n: 256, k: 512, elem_bytes: 2, dtype_id: 2 };

        // Assert: equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Assert: Display format
        assert_eq!(format!("{}", a), "128x256x512_e2_d1");

        // Assert: op_key uses Display format
        let key = op_key("jit_gemm", &a);
        assert_eq!(key, "jit_gemm_128x256x512_e2_d1");
    }

    // ── 10 additional tests (+19 existing = 29 total) ───────────────────────

    #[test]
    fn test_load_nonexistent_file_returns_empty() {
        // Arrange: path that definitely does not exist
        let path = std::env::temp_dir().join("gllm_wisdom_nonexistent_42af9c").join("nope.json");

        // Act
        let db = WisdomDb::load(&path);

        // Assert: empty DB with correct path
        assert_eq!(db.total_entries(), 0);
        assert!(db.fingerprints().is_empty());
        assert_eq!(db.path, path);
    }

    #[test]
    fn test_load_corrupt_json_returns_empty() {
        // Arrange: write garbage to a temp file
        let dir = std::env::temp_dir().join("gllm_wisdom_corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.json");
        std::fs::write(&path, "this is not valid json {{{{").unwrap();

        // Act
        let db = WisdomDb::load(&path);

        // Assert: graceful degradation to empty DB
        assert_eq!(db.total_entries(), 0);
        assert_eq!(db.path, path);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_creates_parent_directory() {
        // Arrange: nested path that does not exist yet
        let dir = std::env::temp_dir().join("gllm_wisdom_mkdir_test_7b2e");
        let _ = std::fs::remove_dir_all(&dir);
        let nested = dir.join("a").join("b").join("c");
        let path = nested.join("wisdom.json");

        let mut db = WisdomDb::new(path.clone());
        let cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };

        // Act: put (makes dirty) then save
        db.put("hw", "op1", cfg, 100.0, Some(5.0));
        db.save().unwrap();

        // Assert: file was created with valid content
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with('{'));
        assert!(content.ends_with('}'));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_get_gemm_blocking_prefers_jit_key_over_legacy() {
        // Arrange: insert both a jit_gemm entry and a legacy gemm entry for the same shape
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let shape = ProblemShape { m: 256, n: 256, k: 256, elem_bytes: 4, dtype_id: 0 };
        let jit_key = op_key("jit_gemm", &shape);
        let std_key = op_key("gemm", &shape);

        let legacy_cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };
        let jit_cfg = TuningConfig { kc: 128, mc: 64, nc: 512, num_threads: 4, jit: None };

        db.put("hw", &std_key, legacy_cfg, 500.0, Some(10.0));
        db.put("hw", &jit_key, jit_cfg, 200.0, Some(50.0));

        // Act
        let result = db.get_gemm_blocking("hw", 256, 256, 256, 4, 0);

        // Assert: returns the jit_gemm entry (kc=128), not the legacy one (kc=64)
        let cached = result.unwrap();
        assert_eq!(cached.config.kc, 128);
        assert!((cached.gflops.unwrap() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_gemm_blocking_dtype_id_isolation() {
        // Arrange: insert a legacy gemm entry (dtype_id=0 implicit)
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let legacy_shape = ProblemShape { m: 512, n: 512, k: 512, elem_bytes: 2, dtype_id: 0 };
        let std_key = op_key("gemm", &legacy_shape);
        let cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };
        db.put("hw", &std_key, cfg, 500.0, None);

        // Act: query with dtype_id != 0 (e.g. BF16=2)
        let result = db.get_gemm_blocking("hw", 512, 512, 512, 2, 2);

        // Assert: non-zero dtype_id must NOT match legacy entries (dtype_id=0 guard)
        assert!(result.is_none());
    }

    #[test]
    fn test_roundtrip_null_gflops() {
        // Arrange: gflops = None (null in JSON)
        let dir = std::env::temp_dir().join("gllm_wisdom_null_gflops");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("null_gflops.json");

        let mut db = WisdomDb::new(path.clone());
        let cfg = TuningConfig { kc: 128, mc: 64, nc: 512, num_threads: 4, jit: None };
        db.put("hw_null", "gemm_64x64x64_e4", cfg, 333.3, None);
        db.save().unwrap();

        // Act: reload
        let db2 = WisdomDb::load(&path);
        let cached = db2.get("hw_null", "gemm_64x64x64_e4").unwrap();

        // Assert: gflops is None, median_ns survives
        assert!(cached.gflops.is_none());
        assert!((cached.median_ns - 333.3).abs() < 0.1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_roundtrip_all_reg_alloc_strategies() {
        // Arrange: roundtrip each RegAllocStrategy variant through serialization
        let dir = std::env::temp_dir().join("gllm_wisdom_all_reg");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("all_reg.json");

        let mut db = WisdomDb::new(path.clone());
        let configs = [
            (RegAllocStrategy::MaxAccumulators, "op_max_acc"),
            (RegAllocStrategy::Balanced, "op_balanced"),
            (RegAllocStrategy::MinSpill, "op_min_spill"),
        ];
        for (strategy, op_name) in &configs {
            let cfg = TuningConfig {
                kc: 64,
                mc: 32,
                nc: 256,
                num_threads: 2,
                jit: Some(JitParams {
                    k_unroll: 4,
                    prefetch_distance: 8,
                    reg_alloc_strategy: *strategy,
                    sw_pipeline_depth: 1,
                    nr_variant: 16,
                }),
            };
            db.put("hw", op_name, cfg, 100.0, None);
        }
        db.save().unwrap();

        // Act: reload
        let db2 = WisdomDb::load(&path);

        // Assert: each strategy survives roundtrip
        for (expected_strategy, op_name) in &configs {
            let cached = db2.get("hw", op_name).unwrap();
            let jit = cached.config.jit.as_ref().unwrap();
            assert_eq!(jit.reg_alloc_strategy, *expected_strategy);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_escape_json_special_characters() {
        // Arrange: strings with backslashes and quotes
        let input = r#"path\with\"quotes\""#;

        // Act
        let escaped = escape_json(input);

        // Assert: backslashes and quotes are escaped
        assert_eq!(escaped, r#"path\\with\\\"quotes\\\""#);
        assert!(!escaped.contains(r#"\"#) || escaped.contains(r#"\\"#));

        // Assert: empty string stays empty
        assert_eq!(escape_json(""), "");
        // Assert: plain ASCII passes through
        assert_eq!(escape_json("hello"), "hello");
    }

    #[test]
    fn test_default_path_returns_non_empty() {
        // Act
        let path = WisdomDb::default_path();

        // Assert: path is non-empty and ends with wisdom.json
        assert!(path.to_string_lossy().len() > 0);
        assert!(
            path.to_string_lossy().ends_with("wisdom.json"),
            "default path should end with wisdom.json, got: {:?}",
            path
        );
    }

    #[test]
    fn test_wisdom_db_save_noop_when_not_dirty() {
        // Arrange: create a path that must NOT be written
        let dir = std::env::temp_dir().join("gllm_wisdom_noop_test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("should_not_exist.json");
        let mut db = WisdomDb::new(path.clone());

        // Act: save without any puts (not dirty)
        db.save().unwrap();

        // Assert: file was never created
        assert!(!path.exists());

        // Act: load an existing file, then save without modifications
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "{}").unwrap();
        let mut db2 = WisdomDb::load(&path);
        db2.save().unwrap();

        // Assert: file content unchanged (still just {})
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content.trim(), "{}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 10 additional tests (+29 existing = 39 total) ───────────────────────

    #[test]
    fn test_split_json_object_empty() {
        // Arrange: empty string (empty JSON object interior)
        let input = "";

        // Act
        let pairs = split_json_object(input);

        // Assert: no key-value pairs extracted
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_split_json_object_single_entry() {
        // Arrange: a single key-value pair
        let input = r#""key1":"value1""#;

        // Act
        let pairs = split_json_object(input);

        // Assert: exactly one pair with correct key and value
        // Note: split_json_object does not strip quotes from values
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "key1");
        assert_eq!(pairs[0].1, "\"value1\"");
    }

    #[test]
    fn test_split_json_object_nested_braces() {
        // Arrange: nested braces inside a value should not split
        let input = r#""outer":"inner{nested}data","next":"val""#;

        // Act
        let pairs = split_json_object(input);

        // Assert: two entries; the nested braces are preserved in the value
        // Note: split_json_object does not strip quotes from values
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "outer");
        assert_eq!(pairs[0].1, "\"inner{nested}data\"");
        assert_eq!(pairs[1].0, "next");
        assert_eq!(pairs[1].1, "\"val\"");
    }

    #[test]
    fn test_extract_usize_missing_key() {
        // Arrange: JSON string that does not contain the key
        let json = r#""other_key":42,"alpha":7"#;

        // Act
        let result = extract_usize(json, "nonexistent");

        // Assert: returns None for missing key
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_f64_negative_value() {
        // Arrange: JSON string with a negative float
        let json = r#""val":-3.14"#;

        // Act
        let result = extract_f64(json, "val");

        // Assert: negative values are parsed correctly
        assert!(result.is_some());
        let val = result.unwrap();
        assert!((val - (-3.14)).abs() < 0.001);
    }

    #[test]
    fn test_extract_f64_scientific_notation() {
        // Arrange: JSON string with scientific notation
        let json = r#""val":1.5e3"#;

        // Act
        let result = extract_f64(json, "val");

        // Assert: scientific notation parsed to correct value
        assert!(result.is_some());
        let val = result.unwrap();
        assert!((val - 1500.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_f64_null_returns_none() {
        // Arrange: JSON string with explicit null value
        let json = r#""val":null"#;

        // Act
        let result = extract_f64(json, "val");

        // Assert: null maps to None
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_wisdom_empty_object() {
        // Arrange: valid but empty JSON object
        let content = "{}";

        // Act
        let result = parse_wisdom(content);

        // Assert: succeeds with empty entries map
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_put_updates_timestamp_monotonically() {
        // Arrange
        let mut db = WisdomDb::new(PathBuf::from("/tmp/_unused.json"));
        let cfg = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };

        // Act: two puts in quick succession
        db.put("hw", "op_ts", cfg.clone(), 100.0, None);
        let ts1 = db.get("hw", "op_ts").unwrap().timestamp;

        // Small sleep to ensure different timestamp
        std::thread::sleep(std::time::Duration::from_millis(10));

        db.put("hw", "op_ts", cfg.clone(), 200.0, None);
        let ts2 = db.get("hw", "op_ts").unwrap().timestamp;

        // Assert: second timestamp >= first
        assert!(ts2 >= ts1, "expected ts2 ({ts2}) >= ts1 ({ts1})");
        // Assert: median_ns was overwritten
        assert!((db.get("hw", "op_ts").unwrap().median_ns - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_all_then_put_and_save_roundtrip() {
        // Arrange: populate DB, clear, then add fresh entry
        let dir = std::env::temp_dir().join("gllm_wisdom_clear_put_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clear_put.json");

        let mut db = WisdomDb::new(path.clone());
        let cfg_old = TuningConfig { kc: 64, mc: 32, nc: 256, num_threads: 2, jit: None };
        db.put("hw", "old_op", cfg_old, 100.0, None);

        // Act: clear, then add a new entry, then save
        db.clear_all();
        let cfg_new = TuningConfig { kc: 256, mc: 128, nc: 1024, num_threads: 8, jit: Some(JitParams::default()) };
        db.put("hw_new", "new_op", cfg_new.clone(), 500.0, Some(60.0));
        db.save().unwrap();

        // Assert: reload has only the new entry
        let db2 = WisdomDb::load(&path);
        assert_eq!(db2.total_entries(), 1);
        assert!(db2.get("hw", "old_op").is_none());
        let cached = db2.get("hw_new", "new_op").unwrap();
        assert_eq!(cached.config.kc, 256);
        assert_eq!(cached.config.jit.as_ref().unwrap().k_unroll, 4);
        assert!((cached.gflops.unwrap() - 60.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
