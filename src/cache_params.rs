//! Runtime cache-aware GEMM blocking parameters.
//!
//! Detects L1D / L2 cache sizes via CPUID (x86) or sysfs (Linux/ARM),
//! then computes optimal KC and MC so that:
//!   - (MR + NR) * KC * sizeof(elem) ≤ L1D * 0.75   (microkernel working set)
//!   - MC * KC * sizeof(elem)         ≤ L2  * 0.50   (A-panel fits half L2)
//!
//! Values are computed once and cached in a static `OnceLock`.

use std::sync::OnceLock;

// ── Cache size detection ─────────────────────────────────────────────

/// (L1D bytes, L2 bytes)
fn detect_cache_sizes() -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(sizes) = detect_x86_cache() {
            return sizes;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(sizes) = detect_sysfs_cache() {
            return sizes;
        }
    }
    // Conservative fallback: 32 KB L1D, 512 KB L2
    (32 * 1024, 512 * 1024)
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_cache() -> Option<(usize, usize)> {
    // CPUID leaf 4: deterministic cache parameters (Intel & AMD Zen+)
    let mut l1d: Option<usize> = None;
    let mut l2: Option<usize> = None;

    for sub in 0..16u32 {
        let info = unsafe { std::arch::x86_64::__cpuid_count(4, sub) };
        let cache_type = info.eax & 0x1F;
        if cache_type == 0 {
            break; // No more caches
        }
        let level = (info.eax >> 5) & 0x7;
        let line_size = (info.ebx & 0xFFF) + 1;
        let partitions = ((info.ebx >> 12) & 0x3FF) + 1;
        let ways = ((info.ebx >> 22) & 0x3FF) + 1;
        let sets = info.ecx + 1;
        let size = line_size as usize * partitions as usize * ways as usize * sets as usize;

        match (level, cache_type) {
            (1, 1) => l1d = Some(size),       // Level 1, Data cache
            (2, 3) | (2, 2) => l2 = Some(size), // Level 2, Unified or Data
            _ => {}
        }
    }

    match (l1d, l2) {
        (Some(d), Some(u)) => Some((d, u)),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn detect_sysfs_cache() -> Option<(usize, usize)> {
    let mut l1d: Option<usize> = None;
    let mut l2: Option<usize> = None;

    for idx in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let level = std::fs::read_to_string(format!("{base}/level")).ok()?;
        let ctype = std::fs::read_to_string(format!("{base}/type")).ok()?;
        let size_str = std::fs::read_to_string(format!("{base}/size")).ok()?;
        let size_str = size_str.trim();

        let size = if let Some(kb) = size_str.strip_suffix('K') {
            kb.parse::<usize>().ok()? * 1024
        } else if let Some(mb) = size_str.strip_suffix('M') {
            mb.parse::<usize>().ok()? * 1024 * 1024
        } else {
            size_str.parse::<usize>().ok()?
        };

        let level: u32 = level.trim().parse().ok()?;
        let ctype = ctype.trim();

        match (level, ctype) {
            (1, "Data") => l1d = Some(size),
            (2, "Unified") => l2 = Some(size),
            _ => {}
        }
    }

    match (l1d, l2) {
        (Some(d), Some(u)) => Some((d, u)),
        _ => None,
    }
}

// ── Blocking parameter computation ───────────────────────────────────

/// Cached blocking parameters for a specific (TM, NV, LANES, elem_size) combo.
#[derive(Debug, Clone, Copy)]
pub struct BlockingParams {
    pub kc: usize,
    pub mc: usize,
}

/// Global cache sizes, detected once.
static CACHE_SIZES: OnceLock<(usize, usize)> = OnceLock::new();

fn cache_sizes() -> (usize, usize) {
    *CACHE_SIZES.get_or_init(detect_cache_sizes)
}

/// Returns the detected L1D size in bytes.
pub fn l1d_size() -> usize {
    cache_sizes().0
}

/// Returns the detected L2 size in bytes.
pub fn l2_size() -> usize {
    cache_sizes().1
}

/// Compute optimal KC given microkernel geometry and element size.
///
/// The B panel (NV*LANES × KC) must stay resident in L1D for the microkernel
/// inner loop. A panel rows are streamed one at a time, so only the B panel
/// dominates the L1D footprint.
///
/// Constraint: NV * LANES * KC * elem_bytes ≤ L1D * 0.80
/// (20% headroom for A streaming + C accumulators in registers)
/// KC is rounded down to a multiple of 8 (for unrolled inner loops) and clamped to [64, 512].
pub fn compute_kc(_tm: usize, nv: usize, lanes: usize, elem_bytes: usize) -> usize {
    let (l1d, _) = cache_sizes();
    let tn = nv * lanes;
    let budget = (l1d * 4) / 5; // 80% of L1D for B panel
    let kc_raw = budget / (tn * elem_bytes);
    // Round down to multiple of 8, clamp to [64, 512]
    let kc = (kc_raw & !7).max(64).min(512);
    kc
}

/// Compute optimal MC given KC, TM, and element size.
///
/// Constraint: MC * KC * elem_bytes ≤ L2 * 0.50
/// MC is rounded down to a multiple of TM and clamped to [TM, 960].
pub fn compute_mc(kc: usize, tm: usize, elem_bytes: usize) -> usize {
    let (_, l2) = cache_sizes();
    let budget = l2 / 2; // 50% of L2
    let mc_raw = budget / (kc * elem_bytes);
    // Round down to multiple of TM, clamp
    let mc = (mc_raw / tm * tm).max(tm).min(960);
    mc
}

/// Compute both KC and MC for a given microkernel geometry.
pub fn blocking_params(tm: usize, nv: usize, lanes: usize, elem_bytes: usize) -> BlockingParams {
    let kc = compute_kc(tm, nv, lanes, elem_bytes);
    let mc = compute_mc(kc, tm, elem_bytes);
    BlockingParams { kc, mc }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_detection() {
        let (l1d, l2) = cache_sizes();
        // Sanity: L1D should be 16KB-128KB, L2 should be 128KB-16MB
        assert!(l1d >= 16 * 1024, "L1D too small: {l1d}");
        assert!(l1d <= 128 * 1024, "L1D too large: {l1d}");
        assert!(l2 >= 128 * 1024, "L2 too small: {l2}");
        assert!(l2 <= 16 * 1024 * 1024, "L2 too large: {l2}");
        eprintln!("Detected: L1D={l1d} bytes, L2={l2} bytes");
    }

    #[test]
    fn test_avx512_f32_params() {
        // AVX-512 f32: TM=14, NV=2, LANES=16, elem=4 bytes
        let p = blocking_params(14, 2, 16, 4);
        eprintln!("AVX-512 f32: KC={}, MC={}", p.kc, p.mc);
        // KC should be reasonable
        assert!(p.kc >= 64 && p.kc <= 512);
        assert!(p.mc >= 14);
        // B panel must fit 80% of L1D: NV*LANES*KC*elem_bytes
        let b_panel = 32 * p.kc * 4;
        assert!(b_panel <= l1d_size() * 4 / 5 + 512, "B panel {b_panel} exceeds L1D budget");
    }

    #[test]
    fn test_avx2_f32_params() {
        // AVX2 f32: TM=6, NV=2, LANES=8, elem=4 bytes
        let p = blocking_params(6, 2, 8, 4);
        eprintln!("AVX2 f32: KC={}, MC={}", p.kc, p.mc);
        assert!(p.kc >= 64 && p.kc <= 512);
        assert!(p.mc >= 6);
    }

    #[test]
    fn test_avx512_bf16_params() {
        // AVX-512 bf16: TM=14, NV=2, LANES=16, elem=2 bytes
        let p = blocking_params(14, 2, 16, 2);
        eprintln!("AVX-512 bf16: KC={}, MC={}", p.kc, p.mc);
        assert!(p.kc >= 64 && p.kc <= 512);
        assert!(p.mc >= 14);
    }
}
