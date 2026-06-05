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

    // ---- 13 additional tests below ----

    #[test]
    fn test_rdtsc_returns_nonzero() {
        // Arrange & Act
        let val = rdtsc();
        // Assert: TSC is always non-zero after boot
        assert_ne!(val, 0, "rdtsc() should never return 0 on a running system");
    }

    #[test]
    fn test_rdtsc_monotonic_three_samples() {
        // Arrange & Act: take three consecutive samples
        let a = rdtsc();
        let b = rdtsc();
        let c = rdtsc();
        // Assert: strict monotonicity (equal is acceptable due to resolution)
        assert!(b >= a, "second sample must be >= first: a={a}, b={b}");
        assert!(c >= b, "third sample must be >= second: b={b}, c={c}");
    }

    #[test]
    fn test_tsc_freq_is_cached_consistently() {
        // Arrange & Act: call tsc_freq_hz() multiple times
        let freq1 = tsc_freq_hz();
        let freq2 = tsc_freq_hz();
        let freq3 = tsc_freq_hz();
        // Assert: OnceLock guarantees the same value every time
        assert_eq!(freq1, freq2, "tsc_freq_hz() must return stable value");
        assert_eq!(freq2, freq3, "tsc_freq_hz() must return stable value");
    }

    #[test]
    fn test_cycles_to_secs_zero() {
        // Arrange
        let cycles = 0u64;
        // Act
        let secs = cycles_to_secs(cycles);
        // Assert
        assert_eq!(secs, 0.0, "0 cycles must convert to 0.0 seconds");
    }

    #[test]
    fn test_cycles_to_ns_zero() {
        // Arrange
        let cycles = 0u64;
        // Act
        let ns = cycles_to_ns(cycles);
        // Assert
        assert_eq!(ns, 0.0, "0 cycles must convert to 0.0 nanoseconds");
    }

    #[test]
    fn test_cycles_to_secs_known_frequency() {
        // Arrange: use the detected frequency to compute expected value
        let freq = tsc_freq_hz();
        // One second worth of cycles
        let one_sec_in_cycles = freq;
        // Act
        let secs = cycles_to_secs(one_sec_in_cycles);
        // Assert: should be approximately 1.0 second (within 1% tolerance for float)
        let diff = (secs - 1.0).abs();
        assert!(diff < 0.01, "freq cycles should be ~1.0s, got {secs}, diff={diff}");
    }

    #[test]
    fn test_cycles_to_ns_known_frequency() {
        // Arrange: 1 million cycles → compute expected ns
        let freq = tsc_freq_hz();
        let cycles = 1_000_000u64;
        let expected_ns = (cycles as f64 * 1e9) / freq as f64;
        // Act
        let ns = cycles_to_ns(cycles);
        // Assert
        let diff = (ns - expected_ns).abs();
        assert!(
            diff < 1.0,
            "cycles_to_ns(1M) = {ns}, expected {expected_ns}, diff={diff}"
        );
    }

    #[test]
    fn test_cycles_to_secs_and_ns_consistency() {
        // Arrange: pick an arbitrary cycle count
        let cycles = 5_000_000u64;
        // Act
        let secs = cycles_to_secs(cycles);
        let ns = cycles_to_ns(cycles);
        // Assert: ns = secs * 1e9
        let ns_from_secs = secs * 1e9;
        let diff = (ns - ns_from_secs).abs();
        assert!(
            diff < 1.0,
            "ns ({ns}) should equal secs*1e9 ({ns_from_secs}), diff={diff}"
        );
    }

    #[test]
    fn test_cycle_timer_default_stop_zero() {
        // Arrange: start a timer but don't stop it
        let timer = CycleTimer::start();
        // Act: check elapsed_cycles on an un-stopped timer
        let cycles = timer.elapsed_cycles();
        // Assert: stop field is 0 (default), so wrapping_sub gives a large number or
        // start itself was 0 — either way, this is a documented "unstopped" state.
        // We just verify it doesn't panic and returns a u64.
        let _ = cycles;
    }

    #[test]
    fn test_cycle_timer_elapsed_positive_after_work() {
        // Arrange
        let mut timer = CycleTimer::start();
        // Act: perform non-trivial work
        let mut acc = 1u64;
        for i in 1..100_000 {
            acc = acc.wrapping_mul(i).wrapping_add(i);
        }
        std::hint::black_box(acc);
        timer.stop();
        // Assert: elapsed must be strictly positive
        assert!(
            timer.elapsed_cycles() > 0,
            "timer must record positive cycles after work"
        );
    }

    #[test]
    fn test_cycle_timer_secs_and_ns_positive() {
        // Arrange
        let mut timer = CycleTimer::start();
        // Act: spin briefly
        let mut v = 0u64;
        for i in 0..50_000 {
            v = v.wrapping_add(i);
        }
        std::hint::black_box(v);
        timer.stop();
        // Assert: both conversions yield positive values
        let secs = timer.elapsed_secs();
        let ns = timer.elapsed_ns();
        assert!(secs > 0.0, "elapsed_secs must be positive, got {secs}");
        assert!(ns > 0.0, "elapsed_ns must be positive, got {ns}");
    }

    #[test]
    fn test_cycle_timer_is_copy() {
        // Arrange: start and stop a timer
        let mut timer = CycleTimer::start();
        let mut x = 0u64;
        for i in 0..1000 {
            x = x.wrapping_add(i);
        }
        std::hint::black_box(x);
        timer.stop();
        let original_cycles = timer.elapsed_cycles();
        // Act: copy the timer
        let copy = timer;
        // Assert: copy has the same state
        assert_eq!(
            copy.elapsed_cycles(),
            original_cycles,
            "copied timer must preserve elapsed cycles"
        );
    }

    #[test]
    fn test_cycle_timer_elapsed_ns_greater_than_elapsed_secs_scaled() {
        // Arrange: measure real work
        let mut timer = CycleTimer::start();
        let mut v = 0u64;
        for i in 0..200_000 {
            v = v.wrapping_add(i * 3);
        }
        std::hint::black_box(v);
        timer.stop();
        // Act
        let secs = timer.elapsed_secs();
        let ns = timer.elapsed_ns();
        // Assert: ns should be roughly secs * 1e9; check same order of magnitude
        let ns_from_secs = secs * 1e9;
        let ratio = if ns_from_secs > 0.0 { ns / ns_from_secs } else { 0.0 };
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "ns ({ns}) / (secs*1e9 = {ns_from_secs}) = {ratio}, expected ~1.0"
        );
    }
}
