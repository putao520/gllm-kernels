//! Runtime cache-aware GEMM blocking parameters.
//!
//! Detects L1D / L2 / L3 cache sizes via CPUID (x86) or sysfs (Linux/ARM),
//! then computes optimal KC, MC, and NC so that:
//!   - (MR + NR) * KC * sizeof(elem) ≤ L1D * 0.75   (microkernel working set)
//!   - MC * KC * sizeof(elem)         ≤ L2  * 0.50   (A-panel fits half L2)
//!   - NC * KC * sizeof(elem)         ≤ L3  * 0.60   (B-panel fits 60% L3)
//!
//! Values are computed once and cached in a static `OnceLock`.

use std::sync::OnceLock;

// ── Cache size detection ─────────────────────────────────────────────

/// (L1D bytes, L2 bytes, L3 bytes)
fn detect_cache_sizes() -> (usize, usize, usize) {
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
    // Conservative fallback: 32 KB L1D, 512 KB L2, 8 MB L3
    (32 * 1024, 512 * 1024, 8 * 1024 * 1024)
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_cache() -> Option<(usize, usize, usize)> {
    // CPUID leaf 4: deterministic cache parameters (Intel & AMD Zen+)
    let mut l1d: Option<usize> = None;
    let mut l2: Option<usize> = None;
    let mut l3: Option<usize> = None;

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
            (3, 3) | (3, 2) => l3 = Some(size), // Level 3, Unified or Data
            _ => {}
        }
    }

    match (l1d, l2) {
        (Some(d), Some(u)) => Some((d, u, l3.unwrap_or(8 * 1024 * 1024))),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn detect_sysfs_cache() -> Option<(usize, usize, usize)> {
    let mut l1d: Option<usize> = None;
    let mut l2: Option<usize> = None;
    let mut l3: Option<usize> = None;

    for idx in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let level = match std::fs::read_to_string(format!("{base}/level")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let ctype = match std::fs::read_to_string(format!("{base}/type")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let size_str = match std::fs::read_to_string(format!("{base}/size")) {
            Ok(s) => s,
            Err(_) => continue,
        };
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
            (3, "Unified") => l3 = Some(size),
            _ => {}
        }
    }

    match (l1d, l2) {
        (Some(d), Some(u)) => Some((d, u, l3.unwrap_or(8 * 1024 * 1024))),
        _ => None,
    }
}

// ── Blocking parameter computation ───────────────────────────────────

/// Cached blocking parameters for a specific (TM, NV, LANES, elem_size) combo.
#[derive(Debug, Clone, Copy)]
pub struct BlockingParams {
    pub kc: usize,
    pub mc: usize,
    pub nc: usize,
}

/// Global cache sizes, detected once.
static CACHE_SIZES: OnceLock<(usize, usize, usize)> = OnceLock::new();

fn cache_sizes() -> (usize, usize, usize) {
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

/// Returns the detected L3 size in bytes.
pub fn l3_size() -> usize {
    cache_sizes().2
}

/// Compute optimal KC given microkernel geometry and element size.
///
/// The microkernel inner loop streams through one TN-wide column of packed B
/// (TN * KC * elem_bytes). The B strip must stay resident in L1D.
/// A elements are streamed (broadcast once per TN FMAs) and don't need L1 residency.
/// C accumulators live in SIMD registers.
///
/// Constraint: TN * KC * elem_bytes ≤ L1D
/// (B strip fills L1D; A streaming + HW prefetch handles the rest)
/// KC is rounded down to a multiple of 8 (for unrolled inner loops) and clamped to [64, 768].
pub fn compute_kc(_tm: usize, nv: usize, lanes: usize, elem_bytes: usize) -> usize {
    let (l1d, _, _) = cache_sizes();
    let tn = nv * lanes;
    let budget = l1d; // Full L1D for B panel strip (A is streamed, C in registers)
    let kc_raw = budget / (tn * elem_bytes);
    // Round down to multiple of 8, clamp to [64, 768]
    let kc = (kc_raw & !7).max(64).min(768);
    kc
}

/// Compute optimal MC given KC, TM, and element size.
///
/// Each thread packs its own A panel (MC × KC) which must fit in L2.
/// Using 80% of L2 leaves room for B panel strips and C tile writes
/// that also transit through L2.
///
/// Constraint: MC * KC * elem_bytes ≤ L2 * 0.80
/// MC is rounded down to a multiple of TM and clamped to [TM, 960].
pub fn compute_mc(kc: usize, tm: usize, elem_bytes: usize) -> usize {
    let (_, l2, _) = cache_sizes();
    let budget = (l2 * 4) / 5; // 80% of L2 for A panel
    let mc_raw = budget / (kc * elem_bytes);
    // Round down to multiple of TM, clamp
    let mc = (mc_raw / tm * tm).max(tm).min(960);
    mc
}

/// Compute optimal NC given KC, TN, and element size.
///
/// The packed B panel (KC × NC) is shared by all cores via L3.
/// Under multi-core contention, effective L3 capacity per core drops,
/// so we use a conservative 40% of L3 to avoid thrashing.
///
/// Constraint: NC * KC * elem_bytes ≤ L3 * 0.40
/// NC is rounded down to a multiple of TN and clamped to [TN, 8192].
pub fn compute_nc(kc: usize, tn: usize, elem_bytes: usize) -> usize {
    let (_, _, l3) = cache_sizes();
    let budget = (l3 * 2) / 5; // 40% of L3 for shared B panel
    let nc_raw = budget / (kc * elem_bytes);
    // Round down to multiple of TN, clamp to [TN, 8192]
    let nc = (nc_raw / tn * tn).max(tn).min(8192);
    nc
}

/// Compute KC, MC, and NC for a given microkernel geometry.
pub fn blocking_params(tm: usize, nv: usize, lanes: usize, elem_bytes: usize) -> BlockingParams {
    let kc = compute_kc(tm, nv, lanes, elem_bytes);
    let mc = compute_mc(kc, tm, elem_bytes);
    let tn = nv * lanes;
    let nc = compute_nc(kc, tn, elem_bytes);
    BlockingParams { kc, mc, nc }
}

// ── Cache-line aligned buffer ─────────────────────────────────────────

/// A `Vec`-like buffer guaranteed to be aligned to 64 bytes (cache line).
/// Used for packed A/B panels so SIMD loads never cross cache-line boundaries.
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> AlignedVec<T> {
    const ALIGN: usize = 64;

    #[inline]
    pub fn new() -> Self {
        Self { ptr: std::ptr::null_mut(), len: 0, cap: 0 }
    }

    #[inline]
    pub fn capacity(&self) -> usize { self.cap }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn as_ptr(&self) -> *const T { self.ptr }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    /// Grow to at least `new_cap` elements, preserving existing data.
    pub fn reserve(&mut self, new_cap: usize) {
        if new_cap <= self.cap { return; }
        let elem_size = std::mem::size_of::<T>();
        assert!(elem_size > 0, "ZST not supported");
        let byte_size = new_cap * elem_size;
        let layout = std::alloc::Layout::from_size_align(byte_size, Self::ALIGN).unwrap();
        let new_ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        assert!(!new_ptr.is_null(), "allocation failed");
        if self.len > 0 && !self.ptr.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len); }
        }
        self.dealloc();
        self.ptr = new_ptr;
        self.cap = new_cap;
    }

    /// Set length without initialization. Caller must ensure elements are valid.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.len = len;
    }

    fn dealloc(&mut self) {
        if !self.ptr.is_null() && self.cap > 0 {
            let elem_size = std::mem::size_of::<T>();
            let byte_size = self.cap * elem_size;
            let layout = std::alloc::Layout::from_size_align(byte_size, Self::ALIGN).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout); }
        }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        self.dealloc();
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_detection() {
        let (l1d, l2, l3) = cache_sizes();
        // Sanity: L1D should be 16KB-128KB, L2 should be 128KB-16MB, L3 should be 1MB-128MB
        assert!(l1d >= 16 * 1024, "L1D too small: {l1d}");
        assert!(l1d <= 128 * 1024, "L1D too large: {l1d}");
        assert!(l2 >= 128 * 1024, "L2 too small: {l2}");
        assert!(l2 <= 16 * 1024 * 1024, "L2 too large: {l2}");
        assert!(l3 >= 1024 * 1024, "L3 too small: {l3}");
        assert!(l3 <= 128 * 1024 * 1024, "L3 too large: {l3}");
        eprintln!("Detected: L1D={l1d} bytes, L2={l2} bytes, L3={l3} bytes");
    }

    #[test]
    fn test_avx512_f32_params() {
        // AVX-512 f32: TM=16, NV=2, LANES=16, elem=4 bytes
        let p = blocking_params(16, 2, 16, 4);
        eprintln!("AVX-512 f32: KC={}, MC={}, NC={}", p.kc, p.mc, p.nc);
        // KC should be reasonable
        assert!(p.kc >= 64 && p.kc <= 768);
        assert!(p.mc >= 16);
        assert!(p.nc >= 32); // at least TN
        // B panel strip must fit in L1D: TN*KC*elem_bytes <= L1D
        let b_panel = 32 * p.kc * 4;
        assert!(b_panel <= l1d_size() + 512, "B panel {b_panel} exceeds L1D");
    }

    #[test]
    fn test_avx2_f32_params() {
        // AVX2 f32: TM=6, NV=2, LANES=8, elem=4 bytes
        let p = blocking_params(6, 2, 8, 4);
        eprintln!("AVX2 f32: KC={}, MC={}, NC={}", p.kc, p.mc, p.nc);
        assert!(p.kc >= 64 && p.kc <= 768);
        assert!(p.mc >= 6);
        assert!(p.nc >= 16); // at least TN
    }

    #[test]
    fn test_avx512_bf16_params() {
        // AVX-512 bf16: TM=16, NV=2, LANES=16, elem=2 bytes
        let p = blocking_params(16, 2, 16, 2);
        eprintln!("AVX-512 bf16: KC={}, MC={}, NC={}", p.kc, p.mc, p.nc);
        assert!(p.kc >= 64 && p.kc <= 768);
        assert!(p.mc >= 16);
        assert!(p.nc >= 32);
    }

    #[test]
    fn test_nc_constraint() {
        // Verify NC * KC * elem_bytes <= L3 * 0.6
        let p = blocking_params(6, 2, 8, 4);
        let tn = 2 * 8;
        let nc_budget = p.nc * p.kc * 4;
        let l3_budget = (l3_size() * 3) / 5;
        assert!(nc_budget <= l3_budget + tn * p.kc * 4,
            "NC budget {nc_budget} exceeds L3 budget {l3_budget}");
        // NC must be multiple of TN
        assert!(p.nc % tn == 0, "NC {} not multiple of TN {}", p.nc, tn);
    }

    #[test]
    fn test_aligned_vec() {
        let mut v = AlignedVec::<f32>::new();
        assert_eq!(v.capacity(), 0);
        v.reserve(1024);
        assert!(v.capacity() >= 1024);
        assert_eq!(v.as_ptr() as usize % 64, 0, "not 64-byte aligned");
        unsafe { v.set_len(1024); }
        // Write and read back
        unsafe {
            for i in 0..1024 { *v.as_mut_ptr().add(i) = i as f32; }
            for i in 0..1024 { assert_eq!(*v.as_ptr().add(i), i as f32); }
        }
        // Re-reserve should preserve data
        v.reserve(2048);
        assert!(v.capacity() >= 2048);
        assert_eq!(v.as_ptr() as usize % 64, 0, "not 64-byte aligned after re-reserve");
        unsafe {
            for i in 0..1024 { assert_eq!(*v.as_ptr().add(i), i as f32); }
        }
    }
}
