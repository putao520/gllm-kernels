//! Parameter cache — FFTW wisdom-style persistence.
//!
//! Tuned parameters are saved to a JSON file keyed by hardware fingerprint
//! and problem shape. On subsequent runs, cached parameters are loaded
//! instantly instead of re-tuning.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::autotuning::search_space::{ProblemShape, TuningConfig};

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
                "    \"{}\": {{\"kc\":{},\"mc\":{},\"nc\":{},\"threads\":{},\"median_ns\":{:.1},\"gflops\":{},\"ts\":{}}}",
                escape_json(op_key),
                r.config.kc,
                r.config.mc,
                r.config.nc,
                r.config.num_threads,
                r.median_ns,
                r.gflops.map(|g| format!("{g:.2}")).unwrap_or_else(|| "null".into()),
                r.timestamp,
            ));
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
) -> Result<HashMap<String, HashMap<String, CachedResult>>, String> {
    // Minimal JSON parser for our known schema.
    // Format: { "hw_fp": { "op_key": { "kc":N, "mc":N, "nc":N, "threads":N, "median_ns":F, "gflops":F|null, "ts":N } } }
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

    Some(CachedResult {
        config: TuningConfig {
            kc,
            mc,
            nc,
            num_threads: threads,
        },
        median_ns,
        gflops,
        timestamp,
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
        assert!((cached.median_ns - 1234.5).abs() < 0.1);
        assert!((cached.gflops.unwrap() - 42.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multiple_entries() {
        let dir = std::env::temp_dir().join("gllm_wisdom_test2");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_wisdom2.json");

        let mut db = WisdomDb::new(path.clone());
        let c1 = TuningConfig { kc: 128, mc: 48, nc: 512, num_threads: 4 };
        let c2 = TuningConfig { kc: 256, mc: 96, nc: 2048, num_threads: 8 };
        db.put("hw1", "gemm_256x256x256_e4", c1, 500.0, Some(10.0));
        db.put("hw1", "gemm_1024x1024x1024_e4", c2, 2000.0, Some(80.0));
        db.put("hw2", "gemm_256x256x256_e4", TuningConfig { kc: 64, mc: 24, nc: 256, num_threads: 2 }, 800.0, None);
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
        let shape = ProblemShape { m: 512, n: 1024, k: 768, elem_bytes: 4 };
        let key = op_key("gemm", &shape);
        assert_eq!(key, "gemm_512x1024x768_e4");
    }
}
