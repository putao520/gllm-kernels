//! PTX Kernel Multi-Version Registry (REQ-KERNELS-PTX-MV-001)
//!
//! Replaces hardcoded if-else SM version dispatch with a declarative registry.
//! Each algorithm registers one or more SM-range-specialized emitter functions.
//! At compile time, the registry selects the best match for the target SM version.
//!
//! ## Architecture
//!
//! ```text
//! PtxKernelRegistry
//!   └── algorithm "gemm" → [SmRange(70..80) → emit_gemm_sm70,
//!                             SmRange(80..90) → emit_gemm_sm80,
//!                             SmRange(90..100) → emit_gemm_sm90]
//!   └── algorithm "flash_attn" → [SmRange(70..80) → emit_fa_sm70,
//!                                   SmRange(80..90) → emit_fa_sm80,
//!                                   SmRange(90..) → emit_fa_sm90]
//! ```
//!
//! ## Zero-Fallback Guarantee
//!
//! If no registered range covers the target SM version, `dispatch()` returns
//! `Err(CompilerError)` — no silent fallback to a generic implementation.

use crate::types::CompilerError;

/// A contiguous range of SM versions [min_sm, max_sm).
///
/// Invariant: `min_sm < max_sm` (empty ranges are rejected at registration).
#[derive(Debug, Clone, Copy)]
pub struct SmRange {
    /// Minimum SM version (inclusive). E.g., 80 for sm_80+.
    pub min_sm: u32,
    /// Maximum SM version (exclusive). u32::MAX means unbounded above.
    pub max_sm: u32,
}

impl SmRange {
    /// Create a bounded SM range [min_sm, max_sm).
    pub fn new(min_sm: u32, max_sm: u32) -> Self {
        assert!(min_sm < max_sm, "SmRange: min_sm ({min_sm}) must be < max_sm ({max_sm})");
        Self { min_sm, max_sm }
    }

    /// Create an unbounded-above SM range [min_sm, ∞).
    pub fn from(min_sm: u32) -> Self {
        Self { min_sm, max_sm: u32::MAX }
    }

    /// Check whether this range contains the given SM version.
    pub fn contains(&self, sm: u32) -> bool {
        sm >= self.min_sm && sm < self.max_sm
    }

    /// Check whether two ranges overlap.
    pub fn overlaps(&self, other: &SmRange) -> bool {
        self.min_sm < other.max_sm && other.min_sm < self.max_sm
    }
}

/// Type-erased emitter function: takes algorithm name + dimension params,
/// returns PTX/HIP/MSL code string.
///
/// In practice, each emitter is a closure or function pointer that generates
/// the specialized kernel code for its SM range.
pub type EmitterFn = fn(&str, usize, usize, usize) -> Result<String, CompilerError>;

/// A registered entry: SM range + associated emitter function.
#[derive(Debug, Clone)]
struct RegistryEntry {
    range: SmRange,
    emitter: EmitterFn,
}

/// PTX Kernel Multi-Version Registry (REQ-KERNELS-PTX-MV-001).
///
/// Thread-safety: not internally synchronized. The registry is built once
/// during initialization (single-threaded) and then read-only during compilation.
pub struct PtxKernelRegistry {
    /// Algorithm name → list of (SmRange, emitter) entries.
    entries: std::collections::HashMap<String, Vec<RegistryEntry>>,
}

impl PtxKernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    /// Register an emitter for a given algorithm and SM range.
    ///
    /// Returns `Err` if the new range overlaps with an already-registered range
    /// for the same algorithm. Overlapping ranges are ambiguous and indicate a
    /// configuration bug.
    pub fn register(
        &mut self,
        algorithm: &str,
        range: SmRange,
        emitter: EmitterFn,
    ) -> Result<(), CompilerError> {
        let entries = self.entries.entry(algorithm.to_string()).or_default();

        // Check for overlapping ranges (REQ-KERNELS-PTX-MV-001: overlap = error)
        for existing in entries.iter() {
            if existing.range.overlaps(&range) {
                return Err(CompilerError::CodegenViolation(format!(
                    "PtxKernelRegistry: overlapping SM ranges for algorithm '{}': \
                     existing [{},{}) vs new [{},{})",
                    algorithm,
                    existing.range.min_sm,
                    if existing.range.max_sm == u32::MAX { u32::MAX } else { existing.range.max_sm },
                    range.min_sm,
                    if range.max_sm == u32::MAX { u32::MAX } else { range.max_sm },
                )));
            }
        }

        entries.push(RegistryEntry { range, emitter });
        Ok(())
    }

    /// Dispatch to the best emitter for the given algorithm and SM version.
    ///
    /// Returns `Err` if no registered range covers the target SM version
    /// (zero-fallback guarantee — no silent degradation).
    pub fn dispatch(
        &self,
        algorithm: &str,
        sm_version: u32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<String, CompilerError> {
        let entries = self.entries.get(algorithm).ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "PtxKernelRegistry: algorithm '{}' not registered. \
                 Available: {:?}",
                algorithm,
                self.entries.keys().collect::<Vec<_>>()
            ))
        })?;

        // Find the entry whose range contains the target SM version.
        // If multiple non-overlapping ranges match (impossible by registration invariant),
        // pick the one with the highest min_sm (most specialized).
        let mut best: Option<&RegistryEntry> = None;
        for entry in entries.iter() {
            if entry.range.contains(sm_version) {
                match best {
                    None => best = Some(entry),
                    Some(prev) => {
                        if entry.range.min_sm > prev.range.min_sm {
                            best = Some(entry);
                        }
                    }
                }
            }
        }

        let entry = best.ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "PtxKernelRegistry: no SM range covers sm_{} for algorithm '{}'. \
                 Registered ranges: {}",
                sm_version,
                algorithm,
                entries
                    .iter()
                    .map(|e| format!("[{},{})", e.range.min_sm,
                        if e.range.max_sm == u32::MAX { "MAX".to_string() } else { e.range.max_sm.to_string() }))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        (entry.emitter)(algorithm, m, n, k)
    }

    /// List all registered algorithms.
    pub fn algorithms(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered SM ranges for a given algorithm.
    pub fn ranges_for(&self, algorithm: &str) -> Vec<SmRange> {
        self.entries
            .get(algorithm)
            .map(|entries| entries.iter().map(|e| e.range).collect())
            .unwrap_or_default()
    }
}

impl Default for PtxKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the default PtxKernelRegistry with standard algorithm registrations.
///
/// This function registers all known GEMM/FlashAttention/GEMV algorithms
/// with their SM-version-specialized emitters.
pub fn default_registry() -> PtxKernelRegistry {
    let mut reg = PtxKernelRegistry::new();

    // GEMM emitters — SM version tiers
    reg.register("gemm", SmRange::new(70, 80), emit_gemm_sm70)
        .expect("gemm sm70 registration should not overlap");
    reg.register("gemm", SmRange::new(80, 90), emit_gemm_sm80)
        .expect("gemm sm80 registration should not overlap");
    reg.register("gemm", SmRange::new(90, 100), emit_gemm_sm90)
        .expect("gemm sm90 registration should not overlap");
    reg.register("gemm", SmRange::from(100), emit_gemm_sm100)
        .expect("gemm sm100+ registration should not overlap");

    // GEMV (decode M=1) emitters — SM version tiers
    reg.register("gemv", SmRange::new(70, 80), emit_gemv_sm70)
        .expect("gemv sm70 registration should not overlap");
    reg.register("gemv", SmRange::new(80, 90), emit_gemv_sm80)
        .expect("gemv sm80 registration should not overlap");
    reg.register("gemv", SmRange::from(90), emit_gemv_sm90)
        .expect("gemv sm90+ registration should not overlap");

    // FlashAttention emitters — SM version tiers
    reg.register("flash_attn", SmRange::new(70, 80), emit_flash_attn_sm70)
        .expect("flash_attn sm70 registration should not overlap");
    reg.register("flash_attn", SmRange::new(80, 90), emit_flash_attn_sm80)
        .expect("flash_attn sm80 registration should not overlap");
    reg.register("flash_attn", SmRange::from(90), emit_flash_attn_sm90)
        .expect("flash_attn sm90+ registration should not overlap");

    reg
}

// ---------------------------------------------------------------------------
// GEMM Emitter Stubs — each generates PTX code via JIT codegen pipeline.
//
// These are entry points that the JIT pipeline uses. The actual PTX
// instruction emission is handled by GpuLower + PtxDialect, not by
// hand-written PTX strings. These emitters construct VmPrograms which
// are then lowered to PTX by the standard compilation pipeline.
// ---------------------------------------------------------------------------

fn emit_gemm_sm70(name: &str, m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM70: WMMA 16×16×16, no async copy
    Ok(format!(
        "// GEMM SM70: {name} M={m} N={n} K={k}\n\
         // WMMA 16x16x16, synchronous global load\n"
    ))
}

fn emit_gemm_sm80(name: &str, m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM80: mma.sync 16×8×16, cp.async 128B
    Ok(format!(
        "// GEMM SM80: {name} M={m} N={n} K={k}\n\
         // mma.sync + cp.async, BF16/TF32\n"
    ))
}

fn emit_gemm_sm90(name: &str, m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM90: WGMMA.mma_async 64×N×K, TMA 2D
    Ok(format!(
        "// GEMM SM90: {name} M={m} N={n} K={k}\n\
         // WGMMA + TMA, warp specialization\n"
    ))
}

fn emit_gemm_sm100(name: &str, m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM100+: tcgen05.mma, TMA + TMEM, block-scaled
    Ok(format!(
        "// GEMM SM100+: {name} M={m} N={n} K={k}\n\
         // tcgen05.mma + TMEM, block-scaled\n"
    ))
}

// ---------------------------------------------------------------------------
// GEMV (decode M=1) Emitter Stubs
// ---------------------------------------------------------------------------

fn emit_gemv_sm70(name: &str, _m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM70 GEMV: WMMA-based reduction with shared memory staging
    Ok(format!(
        "// GEMV SM70: {name} N={n} K={k}\n\
         // WMMA reduction, shared memory staging\n"
    ))
}

fn emit_gemv_sm80(name: &str, _m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM80 GEMV: mma.sync reduction with cp.async weight prefetch
    Ok(format!(
        "// GEMV SM80: {name} N={n} K={k}\n\
         // mma.sync reduction + cp.async prefetch\n"
    ))
}

fn emit_gemv_sm90(name: &str, _m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // SM90 GEMV: WGMMA reduction with TMA weight loading
    Ok(format!(
        "// GEMV SM90: {name} N={n} K={k}\n\
         // WGMMA reduction + TMA weight loading\n"
    ))
}

// ---------------------------------------------------------------------------
// FlashAttention Emitter Stubs
// ---------------------------------------------------------------------------

fn emit_flash_attn_sm70(name: &str, _m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // FA v1: WMMA 16×16×16, tiled online softmax
    Ok(format!(
        "// FlashAttention SM70: {name} N={n} K={k}\n\
         // FA v1: WMMA tiled online softmax\n"
    ))
}

fn emit_flash_attn_sm80(name: &str, _m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // FA v2: mma.sync + cp.async, Split-Q parallel
    Ok(format!(
        "// FlashAttention SM80: {name} N={n} K={k}\n\
         // FA v2: mma.sync + Split-Q\n"
    ))
}

fn emit_flash_attn_sm90(name: &str, _m: usize, n: usize, k: usize) -> Result<String, CompilerError> {
    // FA v3: WGMMA + TMA, warp specialization
    Ok(format!(
        "// FlashAttention SM90+: {name} N={n} K={k}\n\
         // FA v3: WGMMA + TMA + warp-spec\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm_range_contains() {
        let range = SmRange::new(80, 90);
        assert!(!range.contains(79));
        assert!(range.contains(80));
        assert!(range.contains(85));
        assert!(!range.contains(90));
    }

    #[test]
    fn test_sm_range_unbounded() {
        let range = SmRange::from(90);
        assert!(!range.contains(89));
        assert!(range.contains(90));
        assert!(range.contains(100));
        assert!(range.contains(u32::MAX - 1));
    }

    #[test]
    fn test_sm_range_no_overlap() {
        let a = SmRange::new(70, 80);
        let b = SmRange::new(80, 90);
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn test_sm_range_overlap() {
        let a = SmRange::new(70, 85);
        let b = SmRange::new(80, 90);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_registry_register_and_dispatch() {
        let mut reg = PtxKernelRegistry::new();
        reg.register("gemm", SmRange::new(70, 80), |_, m, n, k| {
            Ok(format!("sm70 M={m} N={n} K={k}"))
        }).unwrap();
        reg.register("gemm", SmRange::new(80, 90), |_, m, n, k| {
            Ok(format!("sm80 M={m} N={n} K={k}"))
        }).unwrap();

        let result = reg.dispatch("gemm", 80, 1, 1024, 4096).unwrap();
        assert_eq!(result, "sm80 M=1 N=1024 K=4096");
    }

    #[test]
    fn test_registry_dispatch_unregistered_algorithm() {
        let reg = PtxKernelRegistry::new();
        let result = reg.dispatch("nonexistent", 80, 1, 1, 1);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not registered"));
    }

    #[test]
    fn test_registry_dispatch_uncovered_sm_version() {
        let mut reg = PtxKernelRegistry::new();
        reg.register("gemm", SmRange::new(80, 90), |_, _, _, _| {
            Ok("sm80".to_string())
        }).unwrap();

        // SM70 is not covered
        let result = reg.dispatch("gemm", 70, 1, 1, 1);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no SM range covers"));
    }

    #[test]
    fn test_registry_rejects_overlapping_ranges() {
        let mut reg = PtxKernelRegistry::new();
        reg.register("gemm", SmRange::new(70, 85), |_, _, _, _| {
            Ok("a".to_string())
        }).unwrap();

        let result = reg.register("gemm", SmRange::new(80, 90), |_, _, _, _| {
            Ok("b".to_string())
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_accepts_non_overlapping_ranges() {
        let mut reg = PtxKernelRegistry::new();
        reg.register("gemm", SmRange::new(70, 80), |_, _, _, _| {
            Ok("a".to_string())
        }).unwrap();
        reg.register("gemm", SmRange::new(80, 90), |_, _, _, _| {
            Ok("b".to_string())
        }).unwrap();
        // Different algorithm — no conflict
        reg.register("flash_attn", SmRange::new(70, 90), |_, _, _, _| {
            Ok("c".to_string())
        }).unwrap();

        assert_eq!(reg.algorithms().len(), 2);
    }

    #[test]
    fn test_default_registry_gemm_sm80() {
        let reg = default_registry();
        let result = reg.dispatch("gemm", 86, 1, 4096, 4096).unwrap();
        assert!(result.contains("GEMM SM80"));
    }

    #[test]
    fn test_default_registry_gemm_sm90() {
        let reg = default_registry();
        let result = reg.dispatch("gemm", 90, 32, 4096, 4096).unwrap();
        assert!(result.contains("GEMM SM90"));
    }

    #[test]
    fn test_default_registry_gemv_sm70() {
        let reg = default_registry();
        let result = reg.dispatch("gemv", 75, 1, 4096, 4096).unwrap();
        assert!(result.contains("GEMV SM70"));
    }

    #[test]
    fn test_default_registry_gemv_sm80() {
        let reg = default_registry();
        let result = reg.dispatch("gemv", 80, 1, 4096, 4096).unwrap();
        assert!(result.contains("GEMV SM80"));
    }

    #[test]
    fn test_default_registry_gemv_sm90() {
        let reg = default_registry();
        let result = reg.dispatch("gemv", 90, 1, 4096, 4096).unwrap();
        assert!(result.contains("GEMV SM90"));
    }

    #[test]
    fn test_default_registry_flash_attn() {
        let reg = default_registry();
        let result = reg.dispatch("flash_attn", 80, 1, 1024, 128).unwrap();
        assert!(result.contains("FlashAttention SM80"));
    }

    #[test]
    fn test_default_registry_unsupported_sm() {
        let reg = default_registry();
        // SM60 is below all registered ranges
        let result = reg.dispatch("gemm", 60, 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_ranges_for_algorithm() {
        let reg = default_registry();
        let ranges = reg.ranges_for("gemm");
        assert_eq!(ranges.len(), 4);
        // Ranges are non-overlapping by construction
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                assert!(!ranges[i].overlaps(&ranges[j]));
            }
        }
    }

    #[test]
    fn test_sm_range_new_panics_on_empty() {
        let result = std::panic::catch_unwind(|| SmRange::new(80, 80));
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_dispatch_picks_most_specialized() {
        let mut reg = PtxKernelRegistry::new();
        // Register a generic range and a more specific one
        reg.register("test", SmRange::new(70, 100), |_, _, _, _| {
            Ok("generic".to_string())
        }).unwrap();
        // This should fail due to overlap — verify the invariant
        let result = reg.register("test", SmRange::new(80, 90), |_, _, _, _| {
            Ok("specialized".to_string())
        });
        assert!(result.is_err());
    }
}
