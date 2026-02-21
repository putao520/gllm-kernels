//! Cycle-accurate timing via RDTSC (x86_64) with `Instant` fallback.
//!
//! RDTSC gives sub-nanosecond resolution without kernel transitions.
//! On non-x86 platforms we fall back to `std::time::Instant`.

use std::sync::OnceLock;

/// Read the Time Stamp Counter (cycles since reset).
/// Uses RDTSCP which serializes — no out-of-order measurement skew.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        // RDTSCP: serializing read — waits for all prior instructions to retire.
        // aux (ecx) is discarded; we only need the 64-bit TSC in edx:eax.
        core::arch::asm!(
            "rdtscp",
            out("eax") lo,
            out("edx") hi,
            out("ecx") _,  // processor ID, unused
            options(nostack, nomem),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Serializing fence before measurement start (prevents earlier stores from
/// leaking into the timed region).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn rdtsc_fence() {
    unsafe {
        // LFENCE serializes the instruction stream on Intel (and AMD with
        // MSR C001_1029[1] = 1, which is the default on Zen+).
        core::arch::asm!("lfence", options(nostack, nomem));
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn rdtsc() -> u64 {
    // Fallback: convert Instant to a pseudo-cycle count using estimated frequency.
    let now = std::time::Instant::now();
    // Use elapsed from a fixed epoch so values are monotonically increasing.
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(std::time::Instant::now);
    let nanos = now.duration_since(epoch).as_nanos() as u64;
    // Approximate: assume 3 GHz so 1 ns ~ 3 cycles
    nanos.wrapping_mul(3)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn rdtsc_fence() {
    // No-op on non-x86 — Instant already has sufficient ordering.
}

// ---------------------------------------------------------------------------
// TSC frequency detection
// ---------------------------------------------------------------------------

/// Detected TSC frequency in Hz (cycles per second).
static TSC_FREQ_HZ: OnceLock<u64> = OnceLock::new();

/// Returns the TSC frequency in Hz.
pub fn tsc_freq_hz() -> u64 {
    *TSC_FREQ_HZ.get_or_init(detect_tsc_freq)
}

/// Convert a cycle delta to seconds.
#[inline(always)]
pub fn cycles_to_secs(cycles: u64) -> f64 {
    cycles as f64 / tsc_freq_hz() as f64
}

/// Convert a cycle delta to nanoseconds.
#[inline(always)]
pub fn cycles_to_ns(cycles: u64) -> f64 {
    (cycles as f64 * 1e9) / tsc_freq_hz() as f64
}

fn detect_tsc_freq() -> u64 {
    // Strategy 1 (x86_64): CPUID leaf 0x15 (TSC/core crystal ratio)
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(freq) = detect_tsc_cpuid() {
            return freq;
        }
    }

    // Strategy 2 (Linux): read /proc/cpuinfo for "cpu MHz"
    #[cfg(target_os = "linux")]
    {
        if let Some(freq) = detect_tsc_proc_cpuinfo() {
            return freq;
        }
    }

    // Strategy 3: calibrate by sleeping 10ms and measuring TSC delta
    calibrate_tsc_sleep()
}

#[cfg(target_arch = "x86_64")]
fn detect_tsc_cpuid() -> Option<u64> {
    // Leaf 0x15: eax=denominator, ebx=numerator, ecx=crystal_hz
    // TSC_freq = crystal_hz * numerator / denominator
    let info = core::arch::x86_64::__cpuid(0x15);
    let denom = info.eax as u64;
    let numer = info.ebx as u64;
    let crystal = info.ecx as u64;
    if denom == 0 || numer == 0 {
        return None;
    }
    if crystal != 0 {
        return Some(crystal * numer / denom);
    }
    // Some CPUs report crystal=0 but leaf 0x16 has base frequency in MHz
    let info16 = core::arch::x86_64::__cpuid(0x16);
    let base_mhz = info16.eax as u64 & 0xFFFF;
    if base_mhz > 0 {
        // TSC typically runs at base frequency on modern Intel
        return Some(base_mhz * 1_000_000);
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_tsc_proc_cpuinfo() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        if line.starts_with("cpu MHz") {
            let mhz_str = line.split(':').nth(1)?.trim();
            let mhz: f64 = mhz_str.parse().ok()?;
            return Some((mhz * 1e6) as u64);
        }
    }
    None
}

fn calibrate_tsc_sleep() -> u64 {
    let start = rdtsc();
    let t0 = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let end = rdtsc();
    let elapsed = t0.elapsed();
    let cycles = end.wrapping_sub(start);
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        (cycles as f64 / secs) as u64
    } else {
        3_000_000_000 // fallback: 3 GHz
    }
}

// ---------------------------------------------------------------------------
// High-level timer
// ---------------------------------------------------------------------------

/// A lightweight timer that records start/stop in TSC cycles.
#[derive(Debug, Clone, Copy)]
pub struct CycleTimer {
    start: u64,
    stop: u64,
}

impl CycleTimer {
    /// Start a new timer (records current TSC).
    #[inline(always)]
    pub fn start() -> Self {
        rdtsc_fence();
        let start = rdtsc();
        Self { start, stop: 0 }
    }

    /// Stop the timer (records current TSC).
    #[inline(always)]
    pub fn stop(&mut self) {
        self.stop = rdtsc();
        rdtsc_fence();
    }

    /// Elapsed cycles between start and stop.
    #[inline(always)]
    pub fn elapsed_cycles(&self) -> u64 {
        self.stop.wrapping_sub(self.start)
    }

    /// Elapsed time in seconds.
    #[inline(always)]
    pub fn elapsed_secs(&self) -> f64 {
        cycles_to_secs(self.elapsed_cycles())
    }

    /// Elapsed time in nanoseconds.
    #[inline(always)]
    pub fn elapsed_ns(&self) -> f64 {
        cycles_to_ns(self.elapsed_cycles())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdtsc_monotonic() {
        let a = rdtsc();
        let b = rdtsc();
        assert!(b >= a, "TSC must be monotonic: a={a}, b={b}");
    }

    #[test]
    fn test_tsc_freq_reasonable() {
        let freq = tsc_freq_hz();
        // Should be between 500 MHz and 10 GHz
        assert!(freq >= 500_000_000, "TSC freq too low: {freq}");
        assert!(freq <= 10_000_000_000, "TSC freq too high: {freq}");
        eprintln!("TSC frequency: {:.3} GHz", freq as f64 / 1e9);
    }

    #[test]
    fn test_cycle_timer() {
        let mut timer = CycleTimer::start();
        // Burn some cycles
        let mut x = 0u64;
        for i in 0..10_000 {
            x = x.wrapping_add(i);
        }
        std::hint::black_box(x);
        timer.stop();
        assert!(timer.elapsed_cycles() > 0);
        assert!(timer.elapsed_secs() > 0.0);
        assert!(timer.elapsed_ns() > 0.0);
    }
}
