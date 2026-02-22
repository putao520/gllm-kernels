//! Hardware information detection.
//!
//! Collects CPU topology, cache hierarchy, and ISA features into a single
//! `HwInfo` struct used by `DeviceProfile` for dispatch and tuning decisions.

use std::fmt;

/// Complete hardware fingerprint — uniquely identifies a CPU configuration.
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
    pub neon: bool,
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
        if self.avx512vnni {
            feats.push("VNNI");
        }
        if self.avx2 {
            feats.push("AVX2");
        }
        if self.fma {
            feats.push("FMA");
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
            neon: false,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return IsaFeatures {
            avx2: false,
            fma: false,
            avx512f: false,
            avx512bw: false,
            avx512vnni: false,
            neon: true,
        };
    }
    #[allow(unreachable_code)]
    IsaFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        neon: false,
    }
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
}
