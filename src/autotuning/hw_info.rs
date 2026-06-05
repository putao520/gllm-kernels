//! Hardware information detection for autotuning decisions.
//!
//! Collects CPU topology, cache hierarchy, and ISA features into a single
//! fingerprint used as the cache key for tuned parameters.

use std::fmt;

/// Complete hardware fingerprint — uniquely identifies a CPU configuration
/// so that tuned parameters can be cached and reused across runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HwInfo {
    /// CPU vendor string (e.g. "GenuineIntel", "AuthenticAMD")
    pub vendor: String,
    /// CPU model name (e.g. "Intel(R) Core(TM) i9-13900K")
    pub model_name: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (with HT/SMT)
    pub logical_cores: usize,
    /// L1D cache size in bytes (per core)
    pub l1d_bytes: usize,
    /// L2 cache size in bytes (per core)
    pub l2_bytes: usize,
    /// L3 cache size in bytes (shared)
    pub l3_bytes: usize,
    /// Cache line size in bytes
    pub cacheline_bytes: usize,
    /// Detected ISA level
    pub isa: IsaFeatures,
}

/// ISA feature set relevant to kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsaFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
    /// AVX-512 FP16 (Sapphire Rapids+): native half-precision arithmetic.
    pub avx512fp16: bool,
    /// AVX-512 BF16 (Cooper Lake+): VCVTNE2PS2BF16 / VDPBF16PS instructions.
    pub avx512bf16: bool,
    pub neon: bool,
    /// ARM SVE (Scalable Vector Extension) detected.
    pub sve: bool,
    /// ARM SVE2 detected (superset of SVE).
    pub sve2: bool,
    /// SVE vector length in bytes (runtime-determined via RDVL/CNTB).
    /// 0 if SVE is not available.
    pub sve_vl_bytes: usize,
}

impl fmt::Display for HwInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} | {}P/{}L | L1D={}K L2={}K L3={}M | {}",
            self.model_name,
            self.physical_cores,
            self.logical_cores,
            self.l1d_bytes / 1024,
            self.l2_bytes / 1024,
            self.l3_bytes / (1024 * 1024),
            self.isa,
        )
    }
}

impl fmt::Display for IsaFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut feats = Vec::new();
        if self.avx512f {
            feats.push("AVX-512");
        }
        if self.avx512fp16 {
            feats.push("FP16");
        }
        if self.avx512bf16 {
            feats.push("BF16");
        }
        if self.avx512vnni {
            feats.push("VNNI");
        }
        if self.avx2 {
            feats.push("AVX2");
        }
        if self.fma {
            feats.push("FMA");
        }
        if self.sve2 {
            feats.push("SVE2");
        } else if self.sve {
            feats.push("SVE");
        }
        if self.neon {
            feats.push("NEON");
        }
        if feats.is_empty() {
            feats.push("Scalar");
        }
        write!(f, "{}", feats.join("+"))
    }
}

impl HwInfo {
    /// Detect hardware info for the current machine.
    pub fn detect() -> Self {
        let (l1d, l2, l3) = crate::cache_params::detect_cache_sizes_pub();
        let (vendor, model_name) = detect_cpu_identity();
        let (physical, logical) = detect_core_counts();
        let cacheline = detect_cacheline_size();
        let isa = detect_isa_features();

        HwInfo {
            vendor,
            model_name,
            physical_cores: physical,
            logical_cores: logical,
            l1d_bytes: l1d,
            l2_bytes: l2,
            l3_bytes: l3,
            cacheline_bytes: cacheline,
            isa,
        }
    }

    /// Generate a compact fingerprint string for cache key usage.
    pub fn fingerprint(&self) -> String {
        format!(
            "{}_{}c_l1{}_l2{}_l3{}_{}",
            self.model_name
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>(),
            self.physical_cores,
            self.l1d_bytes / 1024,
            self.l2_bytes / 1024,
            self.l3_bytes / 1024,
            self.isa,
        )
    }
}

// ── CPU identity detection ──────────────────────────────────────────────

fn detect_cpu_identity() -> (String, String) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(id) = detect_x86_identity() {
            return id;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(id) = detect_linux_cpuinfo_identity() {
            return id;
        }
    }
    ("Unknown".into(), "Unknown CPU".into())
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_identity() -> Option<(String, String)> {
    let vendor = {
        let r = std::arch::x86_64::__cpuid(0);
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&r.ebx.to_le_bytes());
        bytes[4..8].copy_from_slice(&r.edx.to_le_bytes());
        bytes[8..12].copy_from_slice(&r.ecx.to_le_bytes());
        String::from_utf8_lossy(&bytes).trim().to_string()
    };

    let model_name = {
        let mut name = String::with_capacity(48);
        for leaf in 0x80000002u32..=0x80000004u32 {
            let r = std::arch::x86_64::__cpuid(leaf);
            for reg in [r.eax, r.ebx, r.ecx, r.edx] {
                for byte in reg.to_le_bytes() {
                    if byte != 0 {
                        name.push(byte as char);
                    }
                }
            }
        }
        name.trim().to_string()
    };

    Some((vendor, model_name))
}

#[cfg(target_os = "linux")]
fn detect_linux_cpuinfo_identity() -> Option<(String, String)> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut vendor = None;
    let mut model = None;
    for line in content.lines() {
        if line.starts_with("vendor_id") {
            vendor = line.split(':').nth(1).map(|s| s.trim().to_string());
        }
        if line.starts_with("model name") {
            model = line.split(':').nth(1).map(|s| s.trim().to_string());
        }
        if vendor.is_some() && model.is_some() {
            break;
        }
    }
    Some((
        vendor.unwrap_or_else(|| "Unknown".into()),
        model.unwrap_or_else(|| "Unknown CPU".into()),
    ))
}

// ── Core count detection ────────────────────────────────────────────────

fn detect_core_counts() -> (usize, usize) {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    #[cfg(target_os = "linux")]
    {
        if let Some(physical) = detect_linux_physical_cores() {
            return (physical, logical);
        }
    }

    // Fallback: assume no HT
    (logical, logical)
}

#[cfg(target_os = "linux")]
fn detect_linux_physical_cores() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut core_ids = std::collections::HashSet::new();
    let mut current_physical = None;
    for line in content.lines() {
        if line.starts_with("physical id") {
            current_physical = line.split(':').nth(1).and_then(|s| s.trim().parse::<usize>().ok());
        }
        if line.starts_with("core id") {
            if let (Some(phys), Some(core)) = (
                current_physical,
                line.split(':').nth(1).and_then(|s| s.trim().parse::<usize>().ok()),
            ) {
                core_ids.insert((phys, core));
            }
        }
    }
    if core_ids.is_empty() {
        None
    } else {
        Some(core_ids.len())
    }
}

// ── Cache line size detection ───────────────────────────────────────────

fn detect_cacheline_size() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let info = std::arch::x86_64::__cpuid(1);
        let cl = ((info.ebx >> 8) & 0xFF) as usize * 8;
        if cl > 0 {
            return cl;
        }
    }
    64 // default
}

// ── ISA feature detection ───────────────────────────────────────────────

fn detect_isa_features() -> IsaFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        return IsaFeatures {
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            avx512f: is_x86_feature_detected!("avx512f"),
            avx512bw: is_x86_feature_detected!("avx512bw"),
            avx512vnni: is_x86_feature_detected!("avx512vnni"),
            avx512fp16: is_x86_feature_detected!("avx512fp16"),
            avx512bf16: is_x86_feature_detected!("avx512bf16"),
            neon: false,
            sve: false,
            sve2: false,
            sve_vl_bytes: 0,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        let sve = std::arch::is_aarch64_feature_detected!("sve");
        let sve2 = std::arch::is_aarch64_feature_detected!("sve2");
        let sve_vl_bytes = if sve { detect_sve_vl_bytes() } else { 0 };
        return IsaFeatures {
            avx2: false,
            fma: false,
            avx512f: false,
            avx512bw: false,
            avx512vnni: false,
            avx512fp16: false,
            avx512bf16: false,
            neon: true,
            sve,
            sve2,
            sve_vl_bytes,
        };
    }
    #[allow(unreachable_code)]
    IsaFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        avx512fp16: false,
        avx512bf16: false,
        neon: false,
        sve: false,
        sve2: false,
        sve_vl_bytes: 0,
    }
}

/// Detect SVE vector length in bytes at runtime.
///
/// On aarch64 with SVE, uses inline assembly (CNTB) to read the
/// hardware vector length. Returns 0 on non-SVE platforms.
#[cfg(target_arch = "aarch64")]
fn detect_sve_vl_bytes() -> usize {
    // CNTB Xd — count the number of bytes in a SVE vector register.
    // This is a safe instruction that reads a system register.
    let vl: u64;
    unsafe {
        std::arch::asm!("cntb {}", out(reg) vl, options(nomem, nostack, preserves_flags));
    }
    vl as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_detect() {
        let hw = HwInfo::detect();
        assert!(hw.physical_cores >= 1);
        assert!(hw.logical_cores >= hw.physical_cores);
        assert!(hw.l1d_bytes >= 16 * 1024);
        assert!(hw.l2_bytes >= 128 * 1024);
        assert!(hw.cacheline_bytes >= 32);
        eprintln!("HwInfo: {hw}");
        eprintln!("Fingerprint: {}", hw.fingerprint());
    }

    #[test]
    fn hw_info_display_format() {
        let hw = HwInfo {
            vendor: "GenuineIntel".into(),
            model_name: "Intel(R) Core(TM) i9-13900K".into(),
            physical_cores: 24,
            logical_cores: 32,
            l1d_bytes: 48 * 1024,
            l2_bytes: 2048 * 1024,
            l3_bytes: 36 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let display = format!("{hw}");
        assert!(display.contains("i9-13900K"));
        assert!(display.contains("24P/32L"));
        assert!(display.contains("AVX2"));
    }

    #[test]
    fn hw_info_fingerprint_format() {
        let hw = HwInfo {
            vendor: "GenuineIntel".into(),
            model_name: "Test CPU v2".into(),
            physical_cores: 8,
            logical_cores: 16,
            l1d_bytes: 32 * 1024,
            l2_bytes: 512 * 1024,
            l3_bytes: 8 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let fp = hw.fingerprint();
        assert!(fp.contains("8c"));
        assert!(fp.contains("AVX2"));
    }

    #[test]
    fn isa_features_display_scalar() {
        let isa = IsaFeatures { avx2: false, fma: false, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 };
        assert_eq!(format!("{isa}"), "Scalar");
    }

    #[test]
    fn isa_features_display_avx512() {
        let isa = IsaFeatures { avx2: true, fma: true, avx512f: true, avx512bw: true, avx512vnni: true, avx512fp16: true, avx512bf16: true, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 };
        let display = format!("{isa}");
        assert!(display.contains("AVX-512"));
        assert!(display.contains("FP16"));
        assert!(display.contains("BF16"));
        assert!(display.contains("VNNI"));
    }

    #[test]
    fn isa_features_display_sve2() {
        let isa = IsaFeatures { avx2: false, fma: false, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: true, sve: true, sve2: true, sve_vl_bytes: 256 };
        let display = format!("{isa}");
        assert!(display.contains("SVE2"));
        assert!(display.contains("NEON"));
    }

    #[test]
    fn hw_info_equality_and_hash() {
        let a = HwInfo {
            vendor: "Test".into(), model_name: "CPU".into(),
            physical_cores: 4, logical_cores: 8,
            l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let b = a.clone();
        assert_eq!(a, b);
        let mut set = std::collections::HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
    }

    // ── 13 new tests ─────────────────────────────────────────────────────

    /// Display cache sizes as KiB / MiB with correct integer division.
    #[test]
    fn hw_info_display_cache_size_units() {
        let hw = HwInfo {
            vendor: "AuthenticAMD".into(),
            model_name: "AMD Ryzen 9 7950X".into(),
            physical_cores: 16,
            logical_cores: 32,
            l1d_bytes: 32 * 1024,          // 32 KiB
            l2_bytes: 1024 * 1024,         // 1024 KiB = 1 MiB
            l3_bytes: 64 * 1024 * 1024,    // 64 MiB
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: true, avx512bw: true, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let display = format!("{hw}");
        // L1D = 32 KiB → "32K"
        assert!(display.contains("L1D=32K"), "L1D should be 32K, got: {display}");
        // L2 = 1024 KiB → "1024K"
        assert!(display.contains("L2=1024K"), "L2 should be 1024K, got: {display}");
        // L3 = 64 MiB → "64M"
        assert!(display.contains("L3=64M"), "L3 should be 64M, got: {display}");
    }

    /// Display shows vendor-derived model name, not the vendor field directly.
    #[test]
    fn hw_info_display_does_not_show_vendor_string() {
        let hw = HwInfo {
            vendor: "GenuineIntel".into(),
            model_name: "Intel(R) Xeon(R) w9-3495X".into(),
            physical_cores: 56,
            logical_cores: 112,
            l1d_bytes: 48 * 1024,
            l2_bytes: 2048 * 1024,
            l3_bytes: 105 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: true, avx512bw: true, avx512vnni: true, avx512fp16: false, avx512bf16: true, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let display = format!("{hw}");
        assert!(display.contains("w9-3495X"), "model name should appear, got: {display}");
        assert!(!display.contains("GenuineIntel"), "vendor field should not appear, got: {display}");
    }

    /// Fingerprint strips non-alphanumeric characters from model name.
    #[test]
    fn fingerprint_strips_non_alphanumeric() {
        let hw = HwInfo {
            vendor: "GenuineIntel".into(),
            model_name: "Intel(R) Core(TM) i7-13700K @ 5.40GHz".into(),
            physical_cores: 16,
            logical_cores: 24,
            l1d_bytes: 48 * 1024,
            l2_bytes: 2048 * 1024,
            l3_bytes: 30 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let fp = hw.fingerprint();
        // Parentheses, hyphens, spaces, @, dots should be stripped
        assert!(!fp.contains('('), "fingerprint should not contain '(': {fp}");
        assert!(!fp.contains(')'), "fingerprint should not contain ')': {fp}");
        assert!(!fp.contains(' '), "fingerprint should not contain spaces: {fp}");
        assert!(!fp.contains('@'), "fingerprint should not contain '@': {fp}");
        assert!(!fp.contains('.'), "fingerprint should not contain '.': {fp}");
        // Core meaningful text should remain
        assert!(fp.contains("IntelRCoreTMi713700K"), "alphanumeric chars should remain: {fp}");
    }

    /// Two HwInfo instances differing only in ISA are not equal.
    #[test]
    fn hw_info_inequality_on_isa_difference() {
        let base = HwInfo {
            vendor: "GenuineIntel".into(),
            model_name: "Test CPU".into(),
            physical_cores: 8,
            logical_cores: 16,
            l1d_bytes: 32 * 1024,
            l2_bytes: 512 * 1024,
            l3_bytes: 8 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: false, fma: false, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let with_avx2 = HwInfo {
            isa: IsaFeatures { avx2: true, fma: false, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
            ..base.clone()
        };
        assert_ne!(base, with_avx2, "HwInfo with different ISA should not be equal");
    }

    /// Two HwInfo differing in physical_cores produce different fingerprints.
    #[test]
    fn fingerprint_differs_on_core_count() {
        let hw_4c = HwInfo {
            vendor: "Test".into(),
            model_name: "CPU-X".into(),
            physical_cores: 4,
            logical_cores: 8,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 4 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let hw_8c = HwInfo {
            physical_cores: 8,
            logical_cores: 16,
            ..hw_4c.clone()
        };
        assert_ne!(hw_4c.fingerprint(), hw_8c.fingerprint());
    }

    /// IsaFeatures Display: SVE shown without SVE2, mutually exclusive.
    #[test]
    fn isa_features_display_sve_without_sve2() {
        let isa = IsaFeatures {
            avx2: false, fma: false,
            avx512f: false, avx512bw: false, avx512vnni: false,
            avx512fp16: false, avx512bf16: false,
            neon: true,
            sve: true, sve2: false, sve_vl_bytes: 128,
        };
        let display = format!("{isa}");
        assert!(display.contains("SVE"), "SVE should appear: {display}");
        assert!(!display.contains("SVE2"), "SVE2 should not appear when sve2=false: {display}");
        assert!(display.contains("NEON"), "NEON should appear: {display}");
    }

    /// IsaFeatures Display: only FMA (no AVX2).
    #[test]
    fn isa_features_display_fma_only() {
        let isa = IsaFeatures {
            avx2: false, fma: true,
            avx512f: false, avx512bw: false, avx512vnni: false,
            avx512fp16: false, avx512bf16: false,
            neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
        };
        let display = format!("{isa}");
        assert!(display.contains("FMA"), "FMA should appear: {display}");
        assert!(!display.contains("AVX2"), "AVX2 should not appear when avx2=false: {display}");
        assert!(!display.contains("Scalar"), "Scalar should not appear when features present: {display}");
    }

    /// IsaFeatures Display: AVX-512 FP16 implies AVX-512 base but not necessarily VNNI.
    #[test]
    fn isa_features_display_avx512fp16_without_vnni() {
        let isa = IsaFeatures {
            avx2: true, fma: true,
            avx512f: true, avx512bw: true, avx512vnni: false,
            avx512fp16: true, avx512bf16: false,
            neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
        };
        let display = format!("{isa}");
        assert!(display.contains("AVX-512"), "AVX-512 should appear: {display}");
        assert!(display.contains("FP16"), "FP16 should appear: {display}");
        assert!(!display.contains("VNNI"), "VNNI should not appear when vnni=false: {display}");
        assert!(!display.contains("BF16"), "BF16 should not appear when bf16=false: {display}");
    }

    /// IsaFeatures equality: sve_vl_bytes matters for equality.
    #[test]
    fn isa_features_inequality_on_sve_vl_bytes() {
        let isa_128 = IsaFeatures {
            avx2: false, fma: false,
            avx512f: false, avx512bw: false, avx512vnni: false,
            avx512fp16: false, avx512bf16: false,
            neon: true, sve: true, sve2: true, sve_vl_bytes: 128,
        };
        let isa_256 = IsaFeatures {
            sve_vl_bytes: 256,
            ..isa_128
        };
        assert_ne!(isa_128, isa_256, "different sve_vl_bytes should not be equal");
    }

    /// IsaFeatures Copy trait: modifying a copy does not affect the original.
    #[test]
    fn isa_features_copy_is_independent() {
        let original = IsaFeatures {
            avx2: true, fma: true,
            avx512f: false, avx512bw: false, avx512vnni: false,
            avx512fp16: false, avx512bf16: false,
            neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
        };
        let mut copy = original;
        copy.avx2 = false;
        copy.avx512f = true;
        assert!(original.avx2, "original should still have avx2=true");
        assert!(!original.avx512f, "original should still have avx512f=false");
        assert!(!copy.avx2, "copy should have avx2=false");
        assert!(copy.avx512f, "copy should have avx512f=true");
    }

    /// HwInfo::detect on the actual machine always returns logical_cores >= physical_cores.
    #[test]
    fn detect_logical_gte_physical_cores() {
        let hw = HwInfo::detect();
        assert!(
            hw.logical_cores >= hw.physical_cores,
            "logical_cores ({}) must be >= physical_cores ({})",
            hw.logical_cores, hw.physical_cores,
        );
    }

    /// HwInfo::detect returns non-empty vendor and model_name strings.
    #[test]
    fn detect_returns_nonempty_identity() {
        let hw = HwInfo::detect();
        assert!(!hw.vendor.is_empty(), "vendor must not be empty");
        assert!(!hw.model_name.is_empty(), "model_name must not be empty");
    }

    /// HwInfo::detect returns positive cache sizes (L1D, L2, L3, cacheline).
    #[test]
    fn detect_returns_positive_cache_sizes() {
        let hw = HwInfo::detect();
        assert!(hw.l1d_bytes > 0, "L1D must be > 0, got {}", hw.l1d_bytes);
        assert!(hw.l2_bytes > 0, "L2 must be > 0, got {}", hw.l2_bytes);
        assert!(hw.l3_bytes > 0, "L3 must be > 0, got {}", hw.l3_bytes);
        assert!(hw.cacheline_bytes > 0, "cacheline must be > 0, got {}", hw.cacheline_bytes);
    }

    // ── 10 additional tests ─────────────────────────────────────────────────

    /// HwInfo Display includes all core count components.
    #[test]
    fn hw_info_display_includes_core_counts() {
        // Arrange
        let hw = HwInfo {
            vendor: "TestVendor".into(),
            model_name: "TestModel".into(),
            physical_cores: 12,
            logical_cores: 24,
            l1d_bytes: 48 * 1024,
            l2_bytes: 512 * 1024,
            l3_bytes: 16 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: false, fma: false, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        // Act
        let display = format!("{hw}");
        // Assert
        assert!(display.contains("12P"), "physical cores should appear: {display}");
        assert!(display.contains("24L"), "logical cores should appear: {display}");
    }

    /// Fingerprint includes cache sizes in KiB units.
    #[test]
    fn fingerprint_includes_cache_sizes_kib() {
        // Arrange
        let hw = HwInfo {
            vendor: "Vendor".into(),
            model_name: "Model".into(),
            physical_cores: 4,
            logical_cores: 8,
            l1d_bytes: 64 * 1024,       // 64 KiB
            l2_bytes: 256 * 1024,       // 256 KiB
            l3_bytes: 8192 * 1024,      // 8192 KiB = 8 MiB
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: false, fma: false, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        // Act
        let fp = hw.fingerprint();
        // Assert
        assert!(fp.contains("l164"), "L1D KiB should appear: {fp}");
        assert!(fp.contains("l2256"), "L2 KiB should appear: {fp}");
        assert!(fp.contains("l38192"), "L3 KiB should appear: {fp}");
    }

    /// HwInfo equality: differing vendor strings means not equal.
    #[test]
    fn hw_info_inequality_on_vendor_difference() {
        // Arrange
        let hw_a = HwInfo {
            vendor: "GenuineIntel".into(),
            model_name: "SameModel".into(),
            physical_cores: 8,
            logical_cores: 16,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 8 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let hw_b = HwInfo {
            vendor: "AuthenticAMD".into(),
            ..hw_a.clone()
        };
        // Act & Assert
        assert_ne!(hw_a, hw_b, "HwInfo with different vendor should not be equal");
    }

    /// HwInfo equality: differing cacheline_bytes means not equal.
    #[test]
    fn hw_info_inequality_on_cacheline_difference() {
        // Arrange
        let hw_64 = HwInfo {
            vendor: "Test".into(),
            model_name: "CPU".into(),
            physical_cores: 4,
            logical_cores: 8,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 4 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let hw_128 = HwInfo {
            cacheline_bytes: 128,
            ..hw_64.clone()
        };
        // Act & Assert
        assert_ne!(hw_64, hw_128, "HwInfo with different cacheline should not be equal");
    }

    /// IsaFeatures Display: AVX-512 BF16 shown without FP16.
    #[test]
    fn isa_features_display_avx512bf16_without_fp16() {
        // Arrange
        let isa = IsaFeatures {
            avx2: true, fma: true,
            avx512f: true, avx512bw: true, avx512vnni: true,
            avx512fp16: false, avx512bf16: true,
            neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
        };
        // Act
        let display = format!("{isa}");
        // Assert
        assert!(display.contains("BF16"), "BF16 should appear: {display}");
        assert!(!display.contains("FP16"), "FP16 should not appear when fp16=false: {display}");
        assert!(display.contains("AVX-512"), "AVX-512 should appear: {display}");
    }

    /// IsaFeatures Display: NEON alone (no SVE/SVE2).
    #[test]
    fn isa_features_display_neon_only() {
        // Arrange
        let isa = IsaFeatures {
            avx2: false, fma: false,
            avx512f: false, avx512bw: false, avx512vnni: false,
            avx512fp16: false, avx512bf16: false,
            neon: true, sve: false, sve2: false, sve_vl_bytes: 0,
        };
        // Act
        let display = format!("{isa}");
        // Assert
        assert!(display.contains("NEON"), "NEON should appear: {display}");
        assert!(!display.contains("SVE"), "SVE should not appear when sve=false: {display}");
        assert!(!display.contains("Scalar"), "Scalar should not appear when NEON present: {display}");
    }

    /// IsaFeatures Display: multiple features joined with '+' separator.
    #[test]
    fn isa_features_display_joins_with_plus() {
        // Arrange
        let isa = IsaFeatures {
            avx2: true, fma: true,
            avx512f: false, avx512bw: false, avx512vnni: false,
            avx512fp16: false, avx512bf16: false,
            neon: false, sve: false, sve2: false, sve_vl_bytes: 0,
        };
        // Act
        let display = format!("{isa}");
        // Assert
        assert!(display.contains("AVX2+FMA"), "features should be joined with '+': {display}");
    }

    /// HwInfo fingerprint: different model names produce different fingerprints.
    #[test]
    fn fingerprint_differs_on_model_name() {
        // Arrange
        let hw_a = HwInfo {
            vendor: "Test".into(),
            model_name: "CPU-Model-A".into(),
            physical_cores: 8,
            logical_cores: 16,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 8 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let hw_b = HwInfo {
            model_name: "CPU-Model-B".into(),
            ..hw_a.clone()
        };
        // Act
        let fp_a = hw_a.fingerprint();
        let fp_b = hw_b.fingerprint();
        // Assert
        assert_ne!(fp_a, fp_b, "fingerprints should differ for different model names");
    }

    /// HwInfo fingerprint: different L3 cache sizes produce different fingerprints.
    #[test]
    fn fingerprint_differs_on_l3_cache() {
        // Arrange
        let hw_small_l3 = HwInfo {
            vendor: "Test".into(),
            model_name: "CPU".into(),
            physical_cores: 8,
            logical_cores: 16,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 4 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let hw_large_l3 = HwInfo {
            l3_bytes: 32 * 1024 * 1024,
            ..hw_small_l3.clone()
        };
        // Act
        let fp_small = hw_small_l3.fingerprint();
        let fp_large = hw_large_l3.fingerprint();
        // Assert
        assert_ne!(fp_small, fp_large, "fingerprints should differ for different L3 sizes");
    }

    /// HwInfo Hash: two equal instances have the same hash.
    #[test]
    fn hw_info_equal_instances_same_hash() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hw = HwInfo {
            vendor: "TestVendor".into(),
            model_name: "TestModel".into(),
            physical_cores: 6,
            logical_cores: 12,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 12 * 1024 * 1024,
            cacheline_bytes: 64,
            isa: IsaFeatures { avx2: true, fma: true, avx512f: false, avx512bw: false, avx512vnni: false, avx512fp16: false, avx512bf16: false, neon: false, sve: false, sve2: false, sve_vl_bytes: 0 },
        };
        let hw_clone = hw.clone();
        // Act
        let mut hasher1 = DefaultHasher::new();
        hw.hash(&mut hasher1);
        let hash1 = hasher1.finish();
        let mut hasher2 = DefaultHasher::new();
        hw_clone.hash(&mut hasher2);
        let hash2 = hasher2.finish();
        // Assert
        assert_eq!(hash1, hash2, "equal HwInfo instances should have the same hash");
    }
}
