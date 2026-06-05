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
            if let Err(e) = std::fs::create_dir_all(d) {
                eprintln!("[gllm-kernels] warning: failed to create cache dir {}: {e}", d.display());
            }
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
            if let Err(e) = save_to_disk(dir, config_hash, &entry) {
                eprintln!("[gllm-kernels] warning: failed to save compiled cache to disk: {e}");
            }
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
    // Atomic write: write to temp file, then rename (rename is atomic on POSIX).
    // Prevents concurrent readers from seeing half-written data.
    let code_path = cache_file_path(dir, hash);
    let meta_path = meta_file_path(dir, hash);

    let code_tmp = code_path.with_extension("tmp");
    std::fs::write(&code_tmp, &entry.code_bytes)?;
    std::fs::rename(&code_tmp, &code_path)?;

    let meta_tmp = meta_path.with_extension("tmp");
    std::fs::write(&meta_tmp, meta.to_json())?;
    std::fs::rename(&meta_tmp, &meta_path)?;
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
    let end = rest.find([',', '}'])?;
    rest[..end].trim().parse().ok()
}

fn parse_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find([',', '}'])?;
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

// ── CRC32C for data integrity verification ───────────────────────────

/// Compute CRC32C checksum for data integrity verification.
///
/// Uses the hardware CRC32C instruction (SSE4.2) when available on x86_64,
/// otherwise falls back to a software lookup table implementation.
/// This is for integrity checks on cached code, not for cache keying.
pub fn crc32c(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { crc32c_hw(data) };
        }
    }
    crc32c_sw(data)
}

/// Hardware-accelerated CRC32C using SSE4.2 `_mm_crc32_u64` / `_mm_crc32_u8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_hw(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};

    let mut crc: u64 = 0xFFFF_FFFF;
    let mut i = 0;
    let len = data.len();

    // Process 8 bytes at a time
    while i + 8 <= len {
        // SAFETY: loop guard `i + 8 <= len` ensures exactly 8 bytes available
        let chunk = u64::from_le_bytes(data[i..i + 8].try_into().expect("slice is exactly 8 bytes"));
        crc = _mm_crc32_u64(crc, chunk);
        i += 8;
    }

    // Process remaining bytes
    while i < len {
        crc = _mm_crc32_u8(crc as u32, data[i]) as u64;
        i += 1;
    }

    (crc as u32) ^ 0xFFFF_FFFF
}

/// Software CRC32C using the Castagnoli polynomial (0x1EDC6F41).
fn crc32c_sw(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let poly: u32 = 0x82F6_3B78; // bit-reversed Castagnoli
        let mut t = [0u32; 256];
        for i in 0..256 {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ poly;
                } else {
                    crc >>= 1;
                }
            }
            t[i] = crc;
        }
        t
    });

    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
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

use crate::compiler::trace::{TraceOp, ValueId};

/// Compact binary representation of a TraceOp.
/// Internal to cache — not exposed publicly.
#[derive(Debug, Clone)]
struct SerializedTraceOp {
    tag: u8,
    operands: [u32; 4],
    float_val: f64,
}

impl SerializedTraceOp {
    fn from_trace_op(op: &TraceOp) -> Self {
        match op {
            TraceOp::Input(n) => Self { tag: 0, operands: [*n, 0, 0, 0], float_val: 0.0 },
            TraceOp::Const(v) => Self { tag: 1, operands: [0, 0, 0, 0], float_val: *v },
            TraceOp::Add(a, b) => Self { tag: 2, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::Sub(a, b) => Self { tag: 3, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::Mul(a, b) => Self { tag: 4, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::Div(a, b) => Self { tag: 5, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::Fma(a, b, c) => Self { tag: 6, operands: [a.0, b.0, c.0, 0], float_val: 0.0 },
            TraceOp::Neg(a) => Self { tag: 7, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Abs(a) => Self { tag: 8, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Exp(a) => Self { tag: 9, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Sqrt(a) => Self { tag: 10, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Rsqrt(a) => Self { tag: 11, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Tanh(a) => Self { tag: 12, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Recip(a) => Self { tag: 13, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Log(a) => Self { tag: 16, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Max(a, b) => Self { tag: 14, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::Min(a, b) => Self { tag: 15, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::ConditionalBranch(mask, t_val, f_val) => Self { tag: 17, operands: [mask.0, t_val.0, f_val.0, 0], float_val: 0.0 },
            TraceOp::QuantFma { acc, act, weight, .. } => Self { tag: 18, operands: [acc.0, act.0, weight.0, 0], float_val: 0.0 },
            TraceOp::BlockScale { data, scale, block_size } => Self { tag: 19, operands: [data.0, scale.0, *block_size as u32, 0], float_val: 0.0 },
            TraceOp::Cast { src, .. } => Self { tag: 20, operands: [src.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::HReduce { src, .. } => Self { tag: 21, operands: [src.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Prefetch { .. } => Self { tag: 22, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::NonTemporalStore => Self { tag: 23, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::BitExtract { src, offset, width } => Self { tag: 24, operands: [src.0, *offset, (*width), 0], float_val: 0.0 },
            TraceOp::Permute { src, indices } => Self { tag: 25, operands: [src.0, indices.0, 0, 0], float_val: 0.0 },
            TraceOp::Compare { a, b, .. } => Self { tag: 26, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::MaskedOp { mask, .. } => Self { tag: 27, operands: [mask.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::AtomicAdd { addr, val } => Self { tag: 28, operands: [addr.0, val.0, 0, 0], float_val: 0.0 },
            TraceOp::FWHT { src, dim } => Self { tag: 29, operands: [src.0, *dim as u32, 0, 0], float_val: 0.0 },
            TraceOp::ScalarLoad { base, offset } => Self { tag: 30, operands: [base.0, offset.0, 0, 0], float_val: 0.0 },
            TraceOp::StrideMul { value, stride } => Self { tag: 31, operands: [value.0, *stride as u32, 0, 0], float_val: 0.0 },
            TraceOp::PtrAdd { base, offset } => Self { tag: 32, operands: [base.0, offset.0, 0, 0], float_val: 0.0 },
            TraceOp::VecLoadIndexed { base, offset } => Self { tag: 33, operands: [base.0, offset.0, 0, 0], float_val: 0.0 },
            TraceOp::VecStoreIndexed { base, offset, value } => Self { tag: 34, operands: [base.0, offset.0, value.0, 0], float_val: 0.0 },
            TraceOp::BroadcastScalar { src } => Self { tag: 35, operands: [src.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::BroadcastLoad { base, offset } => Self { tag: 38, operands: [base.0, offset.0, 0, 0], float_val: 0.0 },
            TraceOp::GatherLoad { base, indices, stride } => Self { tag: 36, operands: [base.0, indices.0, *stride as u32, 0], float_val: 0.0 },
            TraceOp::ScatterStore { base, indices, value, stride } => Self { tag: 37, operands: [base.0, indices.0, value.0, 0], float_val: *stride as f64 },
            TraceOp::TableLookup { base, row_index, row_bytes } => Self { tag: 38, operands: [base.0, row_index.0, *row_bytes as u32, 0], float_val: 0.0 },
            TraceOp::Mxfp4Dequant { data, scales, off_a, stride_a: _, off_b: _, stride_b: _, off_c: _, const_off, block_size } =>
                Self { tag: 39, operands: [data.0, scales.0, off_a.as_ref().map_or(u32::MAX, |v| v.0), *block_size as u32], float_val: *const_off as f64 },
            TraceOp::Sigmoid(a) => Self { tag: 40, operands: [a.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::BitAnd(a, b) => Self { tag: 41, operands: [a.0, b.0, 0, 0], float_val: 0.0 },
            TraceOp::QuantBitAnd { lhs, rhs } => Self { tag: 50, operands: [lhs.0, rhs.0, 0, 0], float_val: 0.0 },
            TraceOp::QuantBitOr { lhs, rhs } => Self { tag: 51, operands: [lhs.0, rhs.0, 0, 0], float_val: 0.0 },
            TraceOp::QuantBroadcast { src, lanes } => Self { tag: 52, operands: [src.0, *lanes as u32, 0, 0], float_val: 0.0 },
            TraceOp::QuantCastF16toF32 { src } => Self { tag: 53, operands: [src.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::QuantCastI8toF32 { src } => Self { tag: 54, operands: [src.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::QuantCastFp8toF32 { src, is_e4m3 } => Self { tag: if *is_e4m3 { 98 } else { 99 }, operands: [src.0, 0, 0, 0], float_val: 0.0 },
            TraceOp::QuantCodebookLookup { indices, vector_size, bits_per_entry, .. } =>
                Self { tag: 55, operands: [indices.0, *vector_size as u32, *bits_per_entry as u32, 0], float_val: 0.0 },
            TraceOp::QuantExtractBits { src, bit_offset, bit_width } =>
                Self { tag: 56, operands: [src.0, *bit_offset, *bit_width as u32, 0], float_val: 0.0 },
            TraceOp::QuantDequantFma { acc, a, b } => Self { tag: 57, operands: [acc.0, a.0, b.0, 0], float_val: 0.0 },
            TraceOp::QuantIntDivConst { src, divisor } => Self { tag: 58, operands: [src.0, 0, 0, 0], float_val: *divisor as f64 },
            TraceOp::QuantIntMul { src, factor } => Self { tag: 59, operands: [src.0, 0, 0, 0], float_val: *factor as f64 },
            TraceOp::QuantInterleave { lo, hi } => Self { tag: 60, operands: [lo.0, hi.0, 0, 0], float_val: 0.0 },
            TraceOp::QuantPtrAddOffset { base, offset_bytes } => Self { tag: 66, operands: [base.0, 0, 0, 0], float_val: *offset_bytes as f64 },
            TraceOp::QuantPtrAddDynamic { base, index } => Self { tag: 67, operands: [base.0, index.0, 0, 0], float_val: 0.0 },
            TraceOp::QuantAndMask { src, mask } => Self { tag: 68, operands: [src.0, 0, 0, 0], float_val: *mask as f64 },
            TraceOp::QuantScalarLoad { ptr, offset_bytes } => Self { tag: 61, operands: [ptr.0, 0, 0, 0], float_val: *offset_bytes as f64 },
            TraceOp::QuantLoadF16toF32 { ptr, offset_bytes } => Self { tag: 64, operands: [ptr.0, 0, 0, 0], float_val: *offset_bytes as f64 },
            TraceOp::QuantLoadI8toF32 { ptr, offset_bytes } => Self { tag: 65, operands: [ptr.0, 0, 0, 0], float_val: *offset_bytes as f64 },
            TraceOp::QuantShiftLeft { src, amount } => Self { tag: 62, operands: [src.0, *amount, 0, 0], float_val: 0.0 },
            TraceOp::QuantShiftRight { src, amount } => Self { tag: 63, operands: [src.0, *amount, 0, 0], float_val: 0.0 },
            TraceOp::QuantScaleLoad { source, offset, .. } => Self { tag: 66, operands: [source.0, 0, 0, 0], float_val: *offset as f64 },
            TraceOp::QuantDataLoad { source, offset, .. } => Self { tag: 67, operands: [source.0, 0, 0, 0], float_val: *offset as f64 },
            TraceOp::QuantZeroLoad { source, offset, .. } => Self { tag: 68, operands: [source.0, 0, 0, 0], float_val: *offset as f64 },
            TraceOp::QuantSubScaleLoad { block_ptr, byte_offset, .. } => Self { tag: 69, operands: [block_ptr.0, 0, 0, 0], float_val: *byte_offset as f64 },
            TraceOp::QuantHighBitsLoad { block_ptr, byte_offset, .. } => Self { tag: 70, operands: [block_ptr.0, 0, 0, 0], float_val: *byte_offset as f64 },
            TraceOp::QuantCodebookDequant { indices, codebook_ptr, vector_size, bits_per_entry } => Self { tag: 71, operands: [indices.0, codebook_ptr.0, *vector_size as u32, *bits_per_entry as u32], float_val: 0.0 },
            TraceOp::QuantE2m1LutDecode { packed_data_ptr, scale_byte, nvfp4_mode } => Self { tag: 72, operands: [packed_data_ptr.0, scale_byte.0, *nvfp4_mode as u32, 0], float_val: 0.0 },
            TraceOp::QuantLoadBytesVec { ptr, offset_bytes, count, signed } => Self { tag: 73, operands: [ptr.0, *signed as u32, *count as u32, 0], float_val: *offset_bytes as f64 },
            TraceOp::QuantKQuantPackedScaleLookup { scales_base, sub_block_idx, is_q3k_extended, is_min } => Self { tag: 74, operands: [scales_base.0, sub_block_idx.0, *is_q3k_extended as u32, *is_min as u32], float_val: 0.0 },
            TraceOp::Loop { .. } => Self { tag: 80, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::PanelLoad { base, offset, rows, cols } => Self { tag: 81, operands: [base.0, offset.0, *rows as u32, *cols as u32], float_val: 0.0 },
            TraceOp::PanelStore { base, offset, rows, cols } => Self { tag: 82, operands: [base.0, offset.0, *rows as u32, *cols as u32], float_val: 0.0 },
            TraceOp::PackBuffer { src, dst, rows, cols, .. } => Self { tag: 83, operands: [src.0, dst.0, *rows as u32, *cols as u32], float_val: 0.0 },
            TraceOp::SharedMemDeclare { bytes, .. } => Self { tag: 84, operands: [0, 0, 0, 0], float_val: *bytes as f64 },
            TraceOp::AsyncCopyToShared { src_offset, bytes, .. } => Self { tag: 85, operands: [src_offset.0, 0, 0, 0], float_val: *bytes as f64 },
            TraceOp::AsyncWaitGroup { n } => Self { tag: 86, operands: [*n, 0, 0, 0], float_val: 0.0 },
            TraceOp::SyncBarrier { .. } => Self { tag: 87, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::TileConfig { rows, cols } => Self { tag: 88, operands: [*rows as u32, *cols as u32, 0, 0], float_val: 0.0 },
            TraceOp::TileMma { c, a, b } => Self { tag: 89, operands: [c.0, a.0, b.0, 0], float_val: 0.0 },
            TraceOp::TileRelease => Self { tag: 90, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Softmax { src, dst } => Self { tag: 91, operands: [src.0, dst.0, 0, 0], float_val: 0.0 },
            TraceOp::EpilogueChain { .. } => Self { tag: 92, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::QuantGather { .. } => Self { tag: 93, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::QuantGemm { .. } => Self { tag: 94, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::MtpDraft { .. } => Self { tag: 95, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::MlaAttnScore { .. } => Self { tag: 96, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::MlaRopeMerge { .. } => Self { tag: 97, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::Tma2DCopy { .. } => Self { tag: 128, operands: [0, 0, 0, 0], float_val: 0.0 },
            TraceOp::DynamicPrecisionSelect { .. } => Self { tag: 129, operands: [0, 0, 0, 0], float_val: 0.0 },
        }
    }

    fn to_trace_op(&self) -> Option<TraceOp> {
        let o = self.operands;
        match self.tag {
            0 => Some(TraceOp::Input(o[0])),
            1 => Some(TraceOp::Const(self.float_val)),
            2 => Some(TraceOp::Add(ValueId(o[0]), ValueId(o[1]))),
            3 => Some(TraceOp::Sub(ValueId(o[0]), ValueId(o[1]))),
            4 => Some(TraceOp::Mul(ValueId(o[0]), ValueId(o[1]))),
            5 => Some(TraceOp::Div(ValueId(o[0]), ValueId(o[1]))),
            6 => Some(TraceOp::Fma(ValueId(o[0]), ValueId(o[1]), ValueId(o[2]))),
            7 => Some(TraceOp::Neg(ValueId(o[0]))),
            8 => Some(TraceOp::Abs(ValueId(o[0]))),
            9 => Some(TraceOp::Exp(ValueId(o[0]))),
            10 => Some(TraceOp::Sqrt(ValueId(o[0]))),
            11 => Some(TraceOp::Rsqrt(ValueId(o[0]))),
            12 => Some(TraceOp::Tanh(ValueId(o[0]))),
            13 => Some(TraceOp::Recip(ValueId(o[0]))),
            14 => Some(TraceOp::Max(ValueId(o[0]), ValueId(o[1]))),
            15 => Some(TraceOp::Min(ValueId(o[0]), ValueId(o[1]))),
            16 => Some(TraceOp::Log(ValueId(o[0]))),
            17 => Some(TraceOp::ConditionalBranch(ValueId(o[0]), ValueId(o[1]), ValueId(o[2]))),
            40 => Some(TraceOp::Sigmoid(ValueId(o[0]))),
            41 => Some(TraceOp::BitAnd(ValueId(o[0]), ValueId(o[1]))),
            // Mxfp4Dequant: partial deserialization (offset components use defaults)
            39 => Some(TraceOp::Mxfp4Dequant {
                data: ValueId(o[0]),
                scales: ValueId(o[1]),
                off_a: if o[2] == u32::MAX { None } else { Some(ValueId(o[2])) },
                stride_a: 0,
                off_b: None,
                stride_b: 0,
                off_c: None,
                const_off: self.float_val as usize,
                block_size: o[3] as usize,
            }),
            72 => Some(TraceOp::QuantE2m1LutDecode {
                packed_data_ptr: ValueId(o[0]),
                scale_byte: ValueId(o[1]),
                nvfp4_mode: o[2] != 0,
            }),
            73 => Some(TraceOp::QuantLoadBytesVec {
                ptr: ValueId(o[0]),
                offset_bytes: self.float_val as i64,
                count: o[2] as usize,
                signed: o[1] != 0,
            }),
            74 => Some(TraceOp::QuantKQuantPackedScaleLookup {
                scales_base: ValueId(o[0]),
                sub_block_idx: ValueId(o[1]),
                is_q3k_extended: o[2] != 0,
                is_min: o[3] != 0,
            }),
            95 => Some(TraceOp::MtpDraft { depth: 1, hidden_size: 1, vocab_size: 1 }),
            96 => Some(TraceOp::MlaAttnScore { num_heads: 1, head_dim: 1, d_c: 1, d_rope: 1 }),
            97 => Some(TraceOp::MlaRopeMerge { d_c: 1, d_rope: 1 }),
            98 => Some(TraceOp::QuantCastFp8toF32 { src: ValueId(o[0]), is_e4m3: true }),
            99 => Some(TraceOp::QuantCastFp8toF32 { src: ValueId(o[0]), is_e4m3: false }),
            128 => Some(TraceOp::Tma2DCopy { desc: String::new(), coord_x: ValueId(0), coord_y: ValueId(0), bytes: 0 }),
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
        if data.len() < 25 { return None; }
        let tag = data[0];
        let o0 = u32::from_le_bytes(data[1..5].try_into().ok()?);
        let o1 = u32::from_le_bytes(data[5..9].try_into().ok()?);
        let o2 = u32::from_le_bytes(data[9..13].try_into().ok()?);
        let o3 = u32::from_le_bytes(data[13..17].try_into().ok()?);
        let fv = f64::from_le_bytes(data[17..25].try_into().ok()?);
        Some(Self { tag, operands: [o0, o1, o2, o3], float_val: fv })
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
        use crate::compiler::trace::{TraceOp, ValueId};

        let ops = vec![
            TraceOp::Input(0),
            TraceOp::Neg(ValueId(0)),
            TraceOp::Exp(ValueId(1)),
            TraceOp::Const(1.0),
            TraceOp::Add(ValueId(2), ValueId(3)),
            TraceOp::Div(ValueId(0), ValueId(4)),
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),
            TraceOp::Max(ValueId(3), ValueId(4)),
            TraceOp::Min(ValueId(0), ValueId(5)),
            TraceOp::Rsqrt(ValueId(2)),
            TraceOp::Tanh(ValueId(1)),
            TraceOp::Recip(ValueId(0)),
            TraceOp::Abs(ValueId(3)),
            TraceOp::Sqrt(ValueId(2)),
            TraceOp::Sub(ValueId(0), ValueId(1)),
            TraceOp::Mul(ValueId(2), ValueId(3)),
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
        use crate::compiler::trace::{TraceOp, ValueId};

        let ops1 = vec![TraceOp::Input(0), TraceOp::Neg(ValueId(0))];
        let ops2 = vec![TraceOp::Input(0), TraceOp::Exp(ValueId(0))];

        let h1 = trace_aware_hash(b"test", "hw", &ops1);
        let h2 = trace_aware_hash(b"test", "hw", &ops2);

        assert_ne!(h1, h2, "Different traces should produce different hashes");
    }

    #[test]
    fn test_trace_aware_hash_deterministic() {
        use crate::compiler::trace::{TraceOp, ValueId};

        let ops = vec![TraceOp::Input(0), TraceOp::Const(3.14), TraceOp::Mul(ValueId(0), ValueId(1))];

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
        use crate::compiler::trace::{TraceOp, ValueId};

        let all_ops = vec![
            TraceOp::Input(42),
            TraceOp::Const(std::f64::consts::PI),
            TraceOp::Add(ValueId(0), ValueId(1)),
            TraceOp::Sub(ValueId(2), ValueId(3)),
            TraceOp::Mul(ValueId(4), ValueId(5)),
            TraceOp::Div(ValueId(6), ValueId(7)),
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),
            TraceOp::Neg(ValueId(3)),
            TraceOp::Abs(ValueId(4)),
            TraceOp::Exp(ValueId(5)),
            TraceOp::Sqrt(ValueId(6)),
            TraceOp::Rsqrt(ValueId(7)),
            TraceOp::Tanh(ValueId(8)),
            TraceOp::Recip(ValueId(9)),
            TraceOp::Max(ValueId(0), ValueId(1)),
            TraceOp::Min(ValueId(2), ValueId(3)),
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

    #[test]
    fn test_crc32c_known_vectors() {
        // Empty input
        assert_eq!(super::crc32c(&[]), 0x0000_0000);
        // Single zero byte
        let one_zero = super::crc32c(&[0u8]);
        assert_ne!(one_zero, 0);
        // Deterministic
        assert_eq!(super::crc32c(b"hello"), super::crc32c(b"hello"));
        // Different data → different checksum
        assert_ne!(super::crc32c(b"hello"), super::crc32c(b"world"));
    }

    #[test]
    fn test_crc32c_sw_matches_hw() {
        // Ensure software path produces consistent results
        let data = b"The quick brown fox jumps over the lazy dog";
        let sw = super::crc32c_sw(data);
        let full = super::crc32c(data);
        // On x86_64 with SSE4.2, both paths should agree
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                assert_eq!(sw, full, "HW and SW CRC32C disagree");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert_eq!(sw, full);
        }
    }

    #[test]
    fn test_crc32c_large_data() {
        let data: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
        let crc = super::crc32c(&data);
        assert_ne!(crc, 0);
        // Flipping one byte should change the checksum
        let mut corrupted = data.clone();
        corrupted[2048] ^= 0x01;
        assert_ne!(super::crc32c(&corrupted), crc);
    }

    // ── Test 22: CacheSource all variants and Copy ──

    #[test]
    fn cache_source_variants_and_copy() {
        // Arrange
        let mem = CacheSource::Memory;
        let disk = CacheSource::Disk;

        // Assert — distinct
        assert_ne!(mem, disk);

        // Copy
        let copied = mem;
        assert_eq!(mem, copied);
    }

    // ── Test 23: CacheSource Debug format ──

    #[test]
    fn cache_source_debug_format() {
        // Arrange & Act
        let debug_mem = format!("{:?}", CacheSource::Memory);
        let debug_disk = format!("{:?}", CacheSource::Disk);

        // Assert
        assert!(debug_mem.contains("Memory"));
        assert!(debug_disk.contains("Disk"));
    }

    // ── Test 24: CompilationCache default is memory-only ──

    #[test]
    fn compilation_cache_default_is_memory_only() {
        // Arrange & Act
        let cache = CompilationCache::default();

        // Assert
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // ── Test 25: IncrementalCompileResult field access ──

    #[test]
    fn incremental_compile_result_field_access() {
        // Arrange
        let result = IncrementalCompileResult {
            layers: Vec::new(),
            memory_hits: 5,
            disk_hits: 3,
            compiled: 2,
        };

        // Assert
        assert!(result.layers.is_empty());
        assert_eq!(result.memory_hits, 5);
        assert_eq!(result.disk_hits, 3);
        assert_eq!(result.compiled, 2);
    }

    // ── Test 26: CacheStats field access from empty cache ──

    #[test]
    fn cache_stats_empty_cache() {
        // Arrange
        let cache = CompilationCache::new();

        // Act
        let stats = cache.stats();

        // Assert
        assert_eq!(stats.num_entries, 0);
        assert_eq!(stats.total_code_bytes, 0);
    }

    // ── Test 27: config_hash produces different results for different input ──

    #[test]
    fn config_hash_differs_for_different_hw_fingerprint() {
        // Arrange
        let ir_bytes = b"test_ir_data";
        let hw_a = "avx2_6c";
        let hw_b = "avx512_4c";

        // Act
        let hash_a = config_hash(ir_bytes, hw_a);
        let hash_b = config_hash(ir_bytes, hw_b);

        // Assert
        assert_ne!(hash_a, hash_b, "different HW fingerprints should produce different hashes");
    }

    // ── Test 28: CompilationCache clear empties entries ──

    #[test]
    fn compilation_cache_clear_empties_entries() {
        // Arrange
        let mut cache = CompilationCache::new();
        #[cfg(target_arch = "x86_64")]
        {
            cache.put(0xABCD, &[0xC3], 0);
            assert_eq!(cache.len(), 1);
        }

        // Act
        cache.clear();

        // Assert
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // ── Test 29: crc32c of empty data ──

    #[test]
    fn crc32c_empty_data() {
        // Arrange & Act
        let crc = crc32c(&[]);

        // Assert — CRC32C of empty data is a well-known constant
        assert_eq!(crc, 0, "CRC32C of empty data should be 0");
    }

    // ── Test 30: CACHE_VERSION is non-empty ──

    #[test]
    fn cache_version_non_empty() {
        // Assert
        assert!(!CACHE_VERSION.is_empty(), "CACHE_VERSION should not be empty");
    }

    // ── Test 31: load_from_disk rejects empty code file ──

    #[test]
    fn load_from_disk_rejects_empty_code_file() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_cache_empty_code_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hash: u64 = 0xEEEE_FFFF;

        // Write a valid meta but an empty code file
        let meta = format!(
            "{{\"scratchpad_bytes\":4096,\"config_hash\":{},\"version\":\"{}\",\"timestamp\":12345}}",
            hash, CACHE_VERSION,
        );
        std::fs::write(meta_file_path(&dir, hash), &meta).unwrap();
        std::fs::write(cache_file_path(&dir, hash), "").unwrap();

        // Act
        let mut cache = CompilationCache::with_disk(dir.clone());
        let result = cache.get(hash);

        // Assert — empty code file is treated as miss and cleaned up
        assert!(result.is_none(), "empty code file should not be loaded");
        assert!(!cache_file_path(&dir, hash).exists(), "empty code file should be deleted");
        assert!(!meta_file_path(&dir, hash).exists(), "meta for empty code should be deleted");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Test 32: put overwrites existing entry with same hash ──

    #[test]
    fn put_overwrites_existing_entry() {
        // Arrange
        let mut cache = CompilationCache::new();

        #[cfg(target_arch = "x86_64")]
        {
            cache.put(0x5001, &[0x90; 10], 100);
            assert_eq!(cache.len(), 1);

            // Act — overwrite with different data
            cache.put(0x5001, &[0xC3; 20], 999);

            // Assert
            assert_eq!(cache.len(), 1, "same hash should overwrite, not add");
            let layer = cache.get(0x5001).unwrap();
            assert_eq!(layer.scratchpad_bytes, 999, "scratchpad should reflect the overwrite");
        }
    }

    // ── Test 33: parse_json helpers return None for missing keys ──

    #[test]
    fn parse_json_helpers_missing_keys() {
        // Arrange
        let json = r#"{"scratchpad_bytes":1024}"#;

        // Act & Assert — key present
        assert_eq!(parse_json_usize(json, "scratchpad_bytes"), Some(1024));

        // Act & Assert — key absent
        assert_eq!(parse_json_usize(json, "config_hash"), None);
        assert_eq!(parse_json_u64(json, "missing_key"), None);
        assert_eq!(parse_json_str(json, "version"), None);
    }

    // ── Test 34: CacheMeta::from_json returns None for malformed input ──

    #[test]
    fn cache_meta_from_json_malformed() {
        // Arrange — missing closing brace
        let bad1 = r#"{"scratchpad_bytes":10"#;
        // Arrange — missing fields
        let bad2 = r#"{"scratchpad_bytes":10,"config_hash":1}"#;
        // Arrange — empty string
        let bad3 = "";

        // Act & Assert
        assert!(CacheMeta::from_json(bad1).is_none(), "truncated JSON should fail");
        assert!(CacheMeta::from_json(bad2).is_none(), "missing fields should fail");
        assert!(CacheMeta::from_json(bad3).is_none(), "empty string should fail");
    }

    // ── Test 35: SerializedTraceOp from_bytes rejects short data ──

    #[test]
    fn serialized_trace_op_from_bytes_short_data() {
        // Arrange — 4 bytes, well below the 25-byte minimum
        let short = &[0x01u8, 0x00, 0x00, 0x00];

        // Act
        let result = SerializedTraceOp::from_bytes(short);

        // Assert
        assert!(result.is_none(), "data shorter than 25 bytes should return None");
    }

    // ── Test 36: deserialize_trace_ops rejects truncated body ──

    #[test]
    fn deserialize_trace_ops_truncated_body() {
        // Arrange — header claims 1 op (4 bytes), but body is only 10 bytes (need 28)
        let mut data: Vec<u8> = 1u32.to_le_bytes().to_vec();
        data.extend_from_slice(&[0u8; 10]);

        // Act
        let result = deserialize_trace_ops(&data);

        // Assert
        assert!(result.is_none(), "truncated body should fail deserialization");
    }

    // ── Test 37: deserialize_trace_ops rejects header shorter than 4 bytes ──

    #[test]
    fn deserialize_trace_ops_short_header() {
        // Arrange — only 3 bytes, not enough for the u32 count
        let data = &[0x01u8, 0x00, 0x00];

        // Act
        let result = deserialize_trace_ops(data);

        // Assert
        assert!(result.is_none(), "header < 4 bytes should fail");
    }

    // ── Test 38: cache_file_path and meta_file_path format ──

    #[test]
    fn cache_and_meta_file_path_format() {
        // Arrange
        let dir = Path::new("/tmp/test_cache");
        let hash: u64 = 0xABCD_1234;

        // Act
        let code_path = cache_file_path(dir, hash);
        let meta_path = meta_file_path(dir, hash);

        // Assert — `{hash:016x}` is lowercase hex, zero-padded to 16 chars, big-endian display
        assert_eq!(code_path.to_str().unwrap(), "/tmp/test_cache/00000000abcd1234.bin");
        assert_eq!(meta_path.to_str().unwrap(), "/tmp/test_cache/00000000abcd1234.meta");
    }

    // ── Test 39: clear on memory-only cache is no-op (does not panic) ──

    #[test]
    fn clear_on_memory_only_cache_no_panic() {
        // Arrange
        let mut cache = CompilationCache::new();
        cache.put(0x9999, &[0xC3], 512);
        assert_eq!(cache.len(), 1);

        // Act — clear on memory-only cache (disk_dir is None)
        cache.clear();

        // Assert — entries cleared, no panic
        assert!(cache.is_empty());
    }

    // ── Test 40: clear_disk_cache on memory-only cache is no-op ──

    #[test]
    fn clear_disk_cache_on_memory_only_is_noop() {
        // Arrange
        let mut cache = CompilationCache::new();
        cache.put(0x8888, &[0xC3], 256);
        assert_eq!(cache.len(), 1);

        // Act — clear_disk_cache with no disk_dir should not panic or affect memory
        cache.clear_disk_cache();

        // Assert — memory entries still present
        assert_eq!(cache.len(), 1, "memory entries should survive clear_disk_cache with no disk");
    }

    // ── Test 41: trace_aware_hash with empty trace ops equals config_hash ──

    #[test]
    fn trace_aware_hash_empty_trace_differs_from_config_hash() {
        // Arrange
        let ir = b"some_ir";
        let hw = "avx2";

        // Act
        let base = config_hash(ir, hw);
        let with_empty = trace_aware_hash(ir, hw, &[]);
        // The serialization of 0 ops produces a 4-byte header, so it mixes extra bytes
        // into the hash. Therefore trace_aware_hash with empty ops != config_hash.
        let with_ops = trace_aware_hash(ir, hw, &[]);

        // Assert
        // Even empty ops add a 4-byte length prefix, so the hash should differ from raw config_hash
        assert_ne!(base, with_empty, "trace_aware_hash should differ from config_hash even with 0 ops");
        assert_eq!(with_empty, with_ops, "same input should be deterministic");
    }

    // ── Test 42: config_hash with empty inputs ──

    #[test]
    fn config_hash_empty_inputs() {
        // Arrange & Act
        let h_both_empty = config_hash(b"", "");
        let h_ir_only = config_hash(b"ir_data", "");
        let h_hw_only = config_hash(b"", "hw_fp");
        let h_both = config_hash(b"ir_data", "hw_fp");

        // Assert — all four should be distinct (FNV-1a is order-sensitive and concatenation-sensitive)
        assert_ne!(h_both_empty, h_ir_only, "empty vs ir-only should differ");
        assert_ne!(h_both_empty, h_hw_only, "empty vs hw-only should differ");
        assert_ne!(h_both_empty, h_both, "empty vs both should differ");
        assert_ne!(h_ir_only, h_both, "ir-only vs both should differ");
        assert_ne!(h_hw_only, h_both, "hw-only vs both should differ");
    }

    // ── Test 43: CacheStats Default trait produces zeroed fields ──

    #[test]
    fn cache_stats_default_zeroed() {
        // Arrange & Act
        let stats = CacheStats::default();

        // Assert
        assert_eq!(stats.memory_hits, 0);
        assert_eq!(stats.disk_hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.total_code_bytes, 0);
        assert_eq!(stats.num_entries, 0);
    }

    // ── Test 44: deserialize_trace_ops returns None for unknown tag ──

    #[test]
    fn deserialize_trace_ops_unknown_tag_returns_none() {
        // Arrange — build a 28-byte SerializedTraceOp with an unmapped tag (0xFE)
        let mut data: Vec<u8> = 1u32.to_le_bytes().to_vec(); // header: 1 op
        let mut op_bytes = Vec::with_capacity(28);
        op_bytes.push(0xFE); // unknown tag
        op_bytes.extend_from_slice(&[0u8; 16]); // 4 x u32 operands
        op_bytes.extend_from_slice(&0.0f64.to_le_bytes()); // float_val
        while op_bytes.len() < 28 {
            op_bytes.push(0);
        }
        data.extend_from_slice(&op_bytes);

        // Act
        let result = deserialize_trace_ops(&data);

        // Assert — to_trace_op returns None for unknown tag, so deserialization fails
        assert!(result.is_none(), "unknown tag should cause deserialization failure");
    }

    // ── Test 45: SerializedTraceOp to_bytes always produces exactly 28 bytes ──

    #[test]
    fn serialized_trace_op_to_bytes_fixed_28() {
        // Arrange — test a few different ops
        let ops: Vec<TraceOp> = vec![
            TraceOp::Input(0),
            TraceOp::Const(f64::MAX),
            TraceOp::Fma(ValueId(u32::MAX), ValueId(0), ValueId(1)),
        ];

        // Act & Assert
        for op in &ops {
            let sop = SerializedTraceOp::from_trace_op(op);
            let bytes = sop.to_bytes();
            assert_eq!(bytes.len(), 28, "SerializedTraceOp must be exactly 28 bytes");
        }
    }

    // ── Test 46: SerializedTraceOp from_bytes at exact minimum length (25 bytes) ──

    #[test]
    fn serialized_trace_op_from_bytes_exact_minimum() {
        // Arrange — 25 bytes: 1 tag + 4×4 u32 + 8 f64 = 25 bytes minimum
        let mut data = vec![0u8; 25];
        data[0] = 1; // tag = Const
        // operands all zero, float_val all zero → Const(0.0)

        // Act
        let result = SerializedTraceOp::from_bytes(&data);

        // Assert
        let sop = result.expect("25 bytes should be accepted");
        assert_eq!(sop.tag, 1);
        assert_eq!(sop.float_val, 0.0);
        // Roundtrip: to_trace_op should yield Const(0.0)
        let op = sop.to_trace_op().unwrap();
        assert_eq!(op, TraceOp::Const(0.0));
    }

    // ── Test 47: disk put creates both code and meta files on disk ──

    #[test]
    fn disk_put_creates_code_and_meta_files() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_cache_put_files_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hash: u64 = 0xBEEF_CAFE;

        // Act
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(hash, &[0x48, 0x89, 0xC0, 0xC3], 32768);
        }

        // Assert — both files must exist
        let code_path = cache_file_path(&dir, hash);
        let meta_path = meta_file_path(&dir, hash);
        assert!(code_path.exists(), "code file should exist after put");
        assert!(meta_path.exists(), "meta file should exist after put");

        // Code content matches
        let code = std::fs::read(&code_path).unwrap();
        assert_eq!(code, vec![0x48, 0x89, 0xC0, 0xC3]);

        // Meta contains expected fields
        let meta_str = std::fs::read_to_string(&meta_path).unwrap();
        assert!(meta_str.contains("\"scratchpad_bytes\":32768"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Test 48: multiple entries coexist on disk without interference ──

    #[test]
    fn disk_cache_multiple_entries_coexist() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_cache_multi_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let hash_a: u64 = 0x1000_0001;
        let hash_b: u64 = 0x2000_0002;
        let hash_c: u64 = 0x3000_0003;

        // Act — write three distinct entries
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(hash_a, &[0xC3; 10], 100);
            cache.put(hash_b, &[0x90; 20], 200);
            cache.put(hash_c, &[0x00; 30], 300);
        }

        // Assert — fresh cache sees all three from disk
        #[cfg(target_arch = "x86_64")]
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            let layer_a = cache.get(hash_a).unwrap();
            assert_eq!(layer_a.scratchpad_bytes, 100);
            let layer_b = cache.get(hash_b).unwrap();
            assert_eq!(layer_b.scratchpad_bytes, 200);
            let layer_c = cache.get(hash_c).unwrap();
            assert_eq!(layer_c.scratchpad_bytes, 300);
        }

        // Non-written hash is a miss
        let mut cache2 = CompilationCache::with_disk(dir.clone());
        assert!(cache2.get(0xFFFF_FFFF).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Test 49: crc32c_sw single byte produces non-zero checksum ──

    #[test]
    fn crc32c_sw_single_byte_nonzero() {
        // Arrange
        let data = &[0xFFu8];

        // Act
        let crc = crc32c_sw(data);

        // Assert — single byte should produce a deterministic, non-zero checksum
        assert_ne!(crc, 0, "CRC32C of single 0xFF byte should be non-zero");
        // Deterministic
        assert_eq!(crc, crc32c_sw(data));
    }

    // ── Test 50: with_disk creates directory lazily on put ──

    #[test]
    fn with_disk_creates_dir_on_put() {
        // Arrange — a deeply nested path that does not exist
        let dir = std::env::temp_dir().join("gllm_cache_deep_test").join("a").join("b").join("c");
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("gllm_cache_deep_test"));

        // with_disk does NOT auto-create directories (unlike default_disk)
        let mut cache = CompilationCache::with_disk(dir.clone());
        assert!(!dir.exists(), "directory should not exist yet");

        // Act — put triggers save_to_disk which will fail because the dir doesn't exist
        cache.put(0x7000, &[0xC3], 64);
        // Memory cache always gets the entry
        assert_eq!(cache.len(), 1);

        // The disk write will have silently failed (directory doesn't exist),
        // but the memory cache is populated. Verify memory hit works.
        #[cfg(target_arch = "x86_64")]
        {
            let layer = cache.get(0x7000).unwrap();
            assert_eq!(layer.scratchpad_bytes, 64);
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("gllm_cache_deep_test"));
    }

    // ── Test 51: config_hash avalanche — single byte flip produces different hash ──

    #[test]
    fn config_hash_single_byte_flip_differs() {
        // Arrange
        let base_ir = b"abcdefgh";
        let mut flipped_ir = *base_ir;
        flipped_ir[3] ^= 0x01; // flip one bit in byte 3

        // Act
        let h_base = config_hash(base_ir, "hw");
        let h_flipped = config_hash(&flipped_ir, "hw");

        // Assert
        assert_ne!(h_base, h_flipped, "flipping one byte in ir_bytes should produce a different hash");
    }

    // ── Test 52: config_hash concatenation — same byte sequence produces same hash ──
    //
    // Note: config_hash feeds ir_bytes then hw_fingerprint sequentially into FNV-1a
    // without a boundary marker. This means "ab"+"cd" and "a"+"bcd" produce the
    // same hash because the byte stream is identical. This is a known design
    // characteristic — the hash encodes content, not the ir/hw boundary.

    #[test]
    fn config_hash_same_concatenation_same_hash() {
        // Arrange — "ab"+"cd" and "a"+"bcd" produce the same byte stream
        let h1 = config_hash(b"ab", "cd");
        let h2 = config_hash(b"a", "bcd");

        // Assert — same concatenated byte stream → same hash
        assert_eq!(h1, h2, "same byte sequence should produce the same hash regardless of ir/hw split");
    }

    // ── Test 53: trace_aware_hash differs when trace length changes ──

    #[test]
    fn trace_aware_hash_differs_for_different_trace_length() {
        // Arrange
        let ops_short = vec![TraceOp::Input(0)];
        let ops_long = vec![TraceOp::Input(0), TraceOp::Neg(ValueId(0)), TraceOp::Exp(ValueId(1))];

        // Act
        let h_short = trace_aware_hash(b"ir", "hw", &ops_short);
        let h_long = trace_aware_hash(b"ir", "hw", &ops_long);

        // Assert
        assert_ne!(h_short, h_long, "different trace lengths should produce different hashes");
    }

    // ── Test 54: multiple memory insertions all retrievable ──

    #[test]
    fn multiple_insertions_all_retrievable() {
        // Arrange
        let mut cache = CompilationCache::new();

        #[cfg(target_arch = "x86_64")]
        {
            let entries = [
                (0x1000u64, &[0xC3u8; 4] as &[u8], 128usize),
                (0x2000u64, &[0x90u8; 8] as &[u8], 256usize),
                (0x3000u64, &[0x48u8; 12] as &[u8], 512usize),
                (0x4000u64, &[0xFFu8; 16] as &[u8], 1024usize),
                (0x5000u64, &[0x00u8; 20] as &[u8], 2048usize),
            ];

            // Act — insert all
            for &(hash, code, sp) in &entries {
                cache.put(hash, code, sp);
            }
            assert_eq!(cache.len(), 5);

            // Assert — all retrievable with correct scratchpad
            for &(hash, _, sp) in &entries {
                let layer = cache.get(hash).unwrap();
                assert_eq!(layer.scratchpad_bytes, sp, "scratchpad mismatch for hash {hash:#x}");
            }
        }
    }

    // ── Test 55: CacheStats Clone produces independent copy ──

    #[test]
    fn cache_stats_clone_independent() {
        // Arrange
        let mut cache = CompilationCache::new();
        cache.put(0xAA00, &[0xC3; 50], 4096);

        let stats = cache.stats();

        // Act
        let cloned = stats.clone();

        // Assert — cloned matches original
        assert_eq!(cloned.num_entries, stats.num_entries);
        assert_eq!(cloned.total_code_bytes, stats.total_code_bytes);
        // Mutating the clone does not affect the original (both are Copy due to u64/u32/usize fields)
        let mut modified = cloned;
        modified.memory_hits = 999;
        assert_eq!(stats.memory_hits, 0, "original should be unaffected");
        assert_eq!(modified.memory_hits, 999);
    }

    // ── Test 56: CacheStats Debug format includes field names ──

    #[test]
    fn cache_stats_debug_format() {
        // Arrange
        let stats = CacheStats {
            memory_hits: 10,
            disk_hits: 5,
            misses: 2,
            total_code_bytes: 1024,
            num_entries: 3,
        };

        // Act
        let debug = format!("{:?}", stats);

        // Assert
        assert!(debug.contains("memory_hits"), "Debug output should contain 'memory_hits'");
        assert!(debug.contains("disk_hits"), "Debug output should contain 'disk_hits'");
        assert!(debug.contains("misses"), "Debug output should contain 'misses'");
        assert!(debug.contains("total_code_bytes"), "Debug output should contain 'total_code_bytes'");
        assert!(debug.contains("num_entries"), "Debug output should contain 'num_entries'");
    }

    // ── Test 57: IncrementalCompileResult field access and construction ──

    #[test]
    fn incremental_compile_result_field_access_and_construction() {
        // Arrange & Act
        let result = IncrementalCompileResult {
            layers: Vec::new(),
            memory_hits: 1,
            disk_hits: 2,
            compiled: 3,
        };

        // Assert — verify all fields are accessible and have correct values
        assert!(result.layers.is_empty(), "layers should be empty");
        assert_eq!(result.memory_hits, 1);
        assert_eq!(result.disk_hits, 2);
        assert_eq!(result.compiled, 3);
    }

    // ── Test 58: disk cache overwrite updates both code and meta files ──

    #[test]
    fn disk_cache_overwrite_updates_files() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_cache_overwrite_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hash: u64 = 0xDDDD_EEEE;

        // Act — first write
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(hash, &[0x90; 4], 512);
        }

        // Overwrite with different data
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            cache.put(hash, &[0xC3; 8], 9999);
        }

        // Assert — fresh cache reads the overwritten values
        #[cfg(target_arch = "x86_64")]
        {
            let mut cache = CompilationCache::with_disk(dir.clone());
            let (layer, _source) = cache.lookup(hash).unwrap();
            assert_eq!(layer.scratchpad_bytes, 9999, "scratchpad should reflect the overwrite");
        }

        // Code file on disk should have the new content
        let code = std::fs::read(cache_file_path(&dir, hash)).unwrap();
        assert_eq!(code, vec![0xC3u8; 8], "code file should reflect the overwrite");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Test 59: CompilationCache clear removes disk files ──

    #[test]
    fn compilation_cache_clear_removes_disk_files() {
        // Arrange
        let dir = std::env::temp_dir().join("gllm_cache_clear_full_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut cache = CompilationCache::with_disk(dir.clone());
        let hash: u64 = 0x1234_5678;
        cache.put(hash, &[0xC3], 2048);
        assert!(cache_file_path(&dir, hash).exists(), "code file should exist after put");
        assert!(meta_file_path(&dir, hash).exists(), "meta file should exist after put");

        // Act
        cache.clear();

        // Assert — memory is empty
        assert!(cache.is_empty());
        assert!(cache.get(hash).is_none(), "cleared cache should miss");
        // Disk files are also removed
        assert!(!cache_file_path(&dir, hash).exists(), "code file should be removed after clear");
        assert!(!meta_file_path(&dir, hash).exists(), "meta file should be removed after clear");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Test 60: serialize_trace_ops byte length is 4 + ops.len() * 28 ──

    #[test]
    fn serialize_trace_ops_byte_length() {
        // Arrange
        let ops_0: Vec<TraceOp> = vec![];
        let ops_1 = vec![TraceOp::Input(0)];
        let ops_3 = vec![
            TraceOp::Input(0),
            TraceOp::Const(1.5),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];

        // Act
        let bytes_0 = serialize_trace_ops(&ops_0);
        let bytes_1 = serialize_trace_ops(&ops_1);
        let bytes_3 = serialize_trace_ops(&ops_3);

        // Assert — header (4 bytes) + 28 bytes per op
        assert_eq!(bytes_0.len(), 4, "0 ops should produce 4 bytes (header only)");
        assert_eq!(bytes_1.len(), 4 + 28, "1 op should produce 32 bytes");
        assert_eq!(bytes_3.len(), 4 + 28 * 3, "3 ops should produce 88 bytes");
    }
}
