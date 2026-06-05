//! Linux `perf_event_open` integration for hardware performance counters.
//!
//! Provides zero-overhead access to:
//! - CPU cycles (actual, not TSC)
//! - Instructions retired
//! - L1D cache misses
//! - LLC (last-level cache) misses
//! - Branch misses
//!
//! On non-Linux platforms, all operations are no-ops returning zero.

/// Raw counter snapshot from perf_event file descriptors.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HwCounters {
    pub cycles: u64,
    pub instructions: u64,
    pub l1d_misses: u64,
    pub llc_misses: u64,
    pub branch_misses: u64,
}

impl HwCounters {
    /// Compute delta: self - baseline.
    pub fn delta(&self, baseline: &HwCounters) -> HwCounters {
        HwCounters {
            cycles: self.cycles.wrapping_sub(baseline.cycles),
            instructions: self.instructions.wrapping_sub(baseline.instructions),
            l1d_misses: self.l1d_misses.wrapping_sub(baseline.l1d_misses),
            llc_misses: self.llc_misses.wrapping_sub(baseline.llc_misses),
            branch_misses: self.branch_misses.wrapping_sub(baseline.branch_misses),
        }
    }

    /// Instructions per cycle.
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 { return 0.0; }
        self.instructions as f64 / self.cycles as f64
    }

    /// L1D miss rate (misses per 1000 instructions).
    pub fn l1d_mpki(&self) -> f64 {
        if self.instructions == 0 { return 0.0; }
        self.l1d_misses as f64 / self.instructions as f64 * 1000.0
    }

    /// LLC miss rate (misses per 1000 instructions).
    pub fn llc_mpki(&self) -> f64 {
        if self.instructions == 0 { return 0.0; }
        self.llc_misses as f64 / self.instructions as f64 * 1000.0
    }
}

// ===========================================================================
// Linux implementation via perf_event_open syscall
// ===========================================================================

#[cfg(target_os = "linux")]
mod linux {
    use super::HwCounters;
    use std::os::unix::io::RawFd;

    // perf_event_open constants (from linux/perf_event.h)
    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
    const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;

    // Cache event encoding
    const PERF_TYPE_HW_CACHE: u32 = 3;
    const PERF_COUNT_HW_CACHE_L1D: u64 = 0;
    const PERF_COUNT_HW_CACHE_LL: u64 = 2;
    const PERF_COUNT_HW_CACHE_OP_READ: u64 = 0;
    const PERF_COUNT_HW_CACHE_RESULT_MISS: u64 = 1;

    #[inline]
    fn cache_event(cache_id: u64, op: u64, result: u64) -> u64 {
        cache_id | (op << 8) | (result << 16)
    }

    /// Minimal perf_event_attr (only fields we need).
    /// The kernel ignores trailing zeros, so we only set the first few fields.
    #[repr(C)]
    #[derive(Default)]
    struct PerfEventAttr {
        type_: u32,
        size: u32,
        config: u64,
        sample_period_or_freq: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64, // bitfield: disabled, inherit, exclude_kernel, exclude_hv, ...
        wakeup_events_or_watermark: u32,
        bp_type: u32,
        config1_or_bp_addr: u64,
        config2_or_bp_len: u64,
        branch_sample_type: u64,
        sample_regs_user: u64,
        sample_stack_user: u32,
        clockid: i32,
        sample_regs_intr: u64,
        aux_watermark: u32,
        sample_max_stack: u16,
        reserved_2: u16,
    }

    fn perf_event_open(attr: &PerfEventAttr, pid: i32, cpu: i32, group_fd: i32, flags: u32) -> i64 {
        unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                attr as *const PerfEventAttr as usize,
                pid,
                cpu,
                group_fd,
                flags,
            )
        }
    }

    fn open_counter(type_: u32, config: u64) -> Option<RawFd> {
        let mut attr = PerfEventAttr::default();
        attr.type_ = type_;
        attr.size = std::mem::size_of::<PerfEventAttr>() as u32;
        attr.config = config;
        // flags: disabled=1, exclude_kernel=1, exclude_hv=1
        // bit 0 = disabled, bit 5 = exclude_kernel, bit 6 = exclude_hv
        attr.flags = (1 << 0) | (1 << 5) | (1 << 6);

        let fd = perf_event_open(&attr, 0, -1, -1, 0);
        if fd < 0 {
            return None;
        }
        Some(fd as RawFd)
    }

    fn read_counter(fd: RawFd) -> u64 {
        let mut val: u64 = 0;
        let ret = unsafe {
            libc::read(fd, &mut val as *mut u64 as *mut libc::c_void, 8)
        };
        if ret == 8 { val } else { 0 }
    }

    fn ioctl_enable(fd: RawFd) {
        // PERF_EVENT_IOC_ENABLE = 0x2400
        unsafe { libc::ioctl(fd, 0x2400, 0); }
    }

    fn ioctl_disable(fd: RawFd) {
        // PERF_EVENT_IOC_DISABLE = 0x2401
        unsafe { libc::ioctl(fd, 0x2401, 0); }
    }

    fn ioctl_reset(fd: RawFd) {
        // PERF_EVENT_IOC_RESET = 0x2403
        unsafe { libc::ioctl(fd, 0x2403, 0); }
    }

    /// Holds open file descriptors for hardware counters.
    pub struct PerfEventGroup {
        fds: Vec<(RawFd, &'static str)>,
    }

    impl PerfEventGroup {
        /// Open all supported counters. Unsupported ones are silently skipped.
        pub fn new() -> Self {
            let mut fds = Vec::new();

            let counters: &[(u32, u64, &str)] = &[
                (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"),
                (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"),
                (PERF_TYPE_HW_CACHE,
                 cache_event(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
                 "l1d_misses"),
                (PERF_TYPE_HW_CACHE,
                 cache_event(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
                 "llc_misses"),
                (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch_misses"),
            ];

            for &(type_, config, name) in counters {
                if let Some(fd) = open_counter(type_, config) {
                    fds.push((fd, name));
                }
            }

            Self { fds }
        }

        /// Returns true if at least one counter was opened.
        pub fn is_available(&self) -> bool {
            !self.fds.is_empty()
        }

        /// Reset and enable all counters.
        pub fn start(&self) {
            for &(fd, _) in &self.fds {
                ioctl_reset(fd);
                ioctl_enable(fd);
            }
        }

        /// Disable all counters and read values.
        pub fn stop(&self) -> HwCounters {
            for &(fd, _) in &self.fds {
                ioctl_disable(fd);
            }
            let mut c = HwCounters::default();
            for &(fd, name) in &self.fds {
                let val = read_counter(fd);
                match name {
                    "cycles" => c.cycles = val,
                    "instructions" => c.instructions = val,
                    "l1d_misses" => c.l1d_misses = val,
                    "llc_misses" => c.llc_misses = val,
                    "branch_misses" => c.branch_misses = val,
                    _ => {}
                }
            }
            c
        }
    }

    impl Drop for PerfEventGroup {
        fn drop(&mut self) {
            for &(fd, _) in &self.fds {
                unsafe { libc::close(fd); }
            }
        }
    }
}

// ===========================================================================
// Non-Linux stub
// ===========================================================================

#[cfg(not(target_os = "linux"))]
mod stub {
    use super::HwCounters;

    pub struct PerfEventGroup;

    impl PerfEventGroup {
        pub fn new() -> Self { Self }
        pub fn is_available(&self) -> bool { false }
        pub fn start(&self) {}
        pub fn stop(&self) -> HwCounters { HwCounters::default() }
    }
}

// ===========================================================================
// Public re-export
// ===========================================================================

#[cfg(target_os = "linux")]
pub use linux::PerfEventGroup;

#[cfg(not(target_os = "linux"))]
pub use stub::PerfEventGroup;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_event_group() {
        let group = PerfEventGroup::new();
        // May or may not be available depending on kernel config / permissions
        eprintln!("perf_event available: {}", group.is_available());

        group.start();
        // Burn some work
        let mut x = 0u64;
        for i in 0..100_000u64 {
            x = x.wrapping_add(i.wrapping_mul(7));
        }
        std::hint::black_box(x);
        let counters = group.stop();

        if group.is_available() {
            eprintln!("cycles:       {}", counters.cycles);
            eprintln!("instructions: {}", counters.instructions);
            eprintln!("IPC:          {:.2}", counters.ipc());
            eprintln!("L1D misses:   {}", counters.l1d_misses);
            eprintln!("LLC misses:   {}", counters.llc_misses);
            eprintln!("L1D MPKI:     {:.2}", counters.l1d_mpki());
        }
    }

    #[test]
    fn test_hw_counters_delta() {
        let a = HwCounters { cycles: 100, instructions: 200, l1d_misses: 5, llc_misses: 1, branch_misses: 3 };
        let b = HwCounters { cycles: 50, instructions: 80, l1d_misses: 2, llc_misses: 0, branch_misses: 1 };
        let d = a.delta(&b);
        assert_eq!(d.cycles, 50);
        assert_eq!(d.instructions, 120);
        assert_eq!(d.l1d_misses, 3);
    }

    #[test]
    fn hw_counters_ipc() {
        let c = HwCounters { cycles: 1000, instructions: 2500, ..Default::default() };
        assert!((c.ipc() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn hw_counters_ipc_zero_cycles() {
        let c = HwCounters::default();
        assert_eq!(c.ipc(), 0.0);
    }

    #[test]
    fn hw_counters_l1d_mpki() {
        let c = HwCounters { cycles: 0, instructions: 1000, l1d_misses: 5, ..Default::default() };
        assert!((c.l1d_mpki() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn hw_counters_llc_mpki() {
        let c = HwCounters { cycles: 0, instructions: 5000, llc_misses: 10, ..Default::default() };
        assert!((c.llc_mpki() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn hw_counters_mpki_zero_instructions() {
        let c = HwCounters { instructions: 0, l1d_misses: 100, llc_misses: 50, ..Default::default() };
        assert_eq!(c.l1d_mpki(), 0.0);
        assert_eq!(c.llc_mpki(), 0.0);
    }

    #[test]
    fn hw_counters_default() {
        let c = HwCounters::default();
        assert_eq!(c.cycles, 0);
        assert_eq!(c.instructions, 0);
        assert_eq!(c.l1d_misses, 0);
        assert_eq!(c.llc_misses, 0);
        assert_eq!(c.branch_misses, 0);
    }

    #[test]
    fn hw_counters_clone_copy() {
        let c = HwCounters { cycles: 42, instructions: 100, l1d_misses: 1, llc_misses: 0, branch_misses: 2 };
        let c2 = c;
        assert_eq!(c2.cycles, 42);
        let c3 = c.clone();
        assert_eq!(c3.instructions, 100);
    }

    #[test]
    fn hw_counters_delta_wrapping() {
        let a = HwCounters { cycles: 10, instructions: 10, l1d_misses: 0, llc_misses: 0, branch_misses: 0 };
        let b = HwCounters { cycles: u64::MAX - 5, instructions: 10, l1d_misses: 0, llc_misses: 0, branch_misses: 0 };
        let d = a.delta(&b);
        // 10 - (u64::MAX - 5) = 10 + 5 + 1 = 16 (wrapping)
        assert_eq!(d.cycles, 16);
    }

    #[test]
    fn hw_counters_delta_all_fields() {
        let a = HwCounters { cycles: 500, instructions: 1000, l1d_misses: 20, llc_misses: 8, branch_misses: 15 };
        let b = HwCounters { cycles: 100, instructions: 200, l1d_misses: 5, llc_misses: 2, branch_misses: 3 };
        let d = a.delta(&b);
        assert_eq!(d.cycles, 400);
        assert_eq!(d.instructions, 800);
        assert_eq!(d.l1d_misses, 15);
        assert_eq!(d.llc_misses, 6);
        assert_eq!(d.branch_misses, 12);
    }

    #[test]
    fn hw_counters_delta_self_is_zero() {
        let c = HwCounters { cycles: 999, instructions: 888, l1d_misses: 77, llc_misses: 66, branch_misses: 55 };
        let d = c.delta(&c);
        assert_eq!(d.cycles, 0);
        assert_eq!(d.instructions, 0);
        assert_eq!(d.l1d_misses, 0);
        assert_eq!(d.llc_misses, 0);
        assert_eq!(d.branch_misses, 0);
    }

    #[test]
    fn hw_counters_delta_default_baseline() {
        let c = HwCounters { cycles: 100, instructions: 200, l1d_misses: 3, llc_misses: 1, branch_misses: 2 };
        let d = c.delta(&HwCounters::default());
        assert_eq!(d.cycles, c.cycles);
        assert_eq!(d.instructions, c.instructions);
        assert_eq!(d.l1d_misses, c.l1d_misses);
        assert_eq!(d.llc_misses, c.llc_misses);
        assert_eq!(d.branch_misses, c.branch_misses);
    }

    #[test]
    fn hw_counters_debug_format() {
        let c = HwCounters { cycles: 42, instructions: 100, l1d_misses: 5, llc_misses: 1, branch_misses: 3 };
        let s = format!("{:?}", c);
        assert!(s.contains("cycles: 42"), "Debug output should contain cycles field");
        assert!(s.contains("instructions: 100"), "Debug output should contain instructions field");
        assert!(s.contains("l1d_misses: 5"), "Debug output should contain l1d_misses field");
        assert!(s.contains("llc_misses: 1"), "Debug output should contain llc_misses field");
        assert!(s.contains("branch_misses: 3"), "Debug output should contain branch_misses field");
    }

    #[test]
    fn hw_counters_equality() {
        let a = HwCounters { cycles: 10, instructions: 20, l1d_misses: 1, llc_misses: 2, branch_misses: 3 };
        let b = HwCounters { cycles: 10, instructions: 20, l1d_misses: 1, llc_misses: 2, branch_misses: 3 };
        let c = HwCounters { cycles: 10, instructions: 20, l1d_misses: 1, llc_misses: 2, branch_misses: 99 };
        // Verify field-by-field equality for identical values
        assert_eq!(a.cycles, b.cycles);
        assert_eq!(a.instructions, b.instructions);
        assert_eq!(a.l1d_misses, b.l1d_misses);
        assert_eq!(a.llc_misses, b.llc_misses);
        assert_eq!(a.branch_misses, b.branch_misses);
        // Verify inequality when one field differs
        assert_ne!(a.branch_misses, c.branch_misses);
    }

    #[test]
    fn hw_counters_max_values() {
        let c = HwCounters { cycles: u64::MAX, instructions: u64::MAX, l1d_misses: u64::MAX, llc_misses: u64::MAX, branch_misses: u64::MAX };
        assert_eq!(c.cycles, u64::MAX);
        assert_eq!(c.instructions, u64::MAX);
        // IPC with max values should be 1.0 (equal numerator and denominator)
        assert!((c.ipc() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hw_counters_ipc_high_ratio() {
        let c = HwCounters { cycles: 1, instructions: u64::MAX, ..Default::default() };
        let ipc = c.ipc();
        assert!(ipc > 0.0, "IPC should be positive when instructions > 0 and cycles > 0");
        assert!(ipc.is_finite(), "IPC should be finite for valid inputs");
    }

    #[test]
    fn hw_counters_mpki_scaling() {
        // 50 misses per 10000 instructions = 5.0 MPKI
        let c = HwCounters { instructions: 10000, l1d_misses: 50, llc_misses: 25, ..Default::default() };
        assert!((c.l1d_mpki() - 5.0).abs() < 1e-6);
        assert!((c.llc_mpki() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn hw_counters_clone_independence() {
        let mut c = HwCounters { cycles: 100, instructions: 200, l1d_misses: 5, llc_misses: 2, branch_misses: 1 };
        let c2 = c.clone();
        c.cycles = 999;
        assert_eq!(c2.cycles, 100, "Clone should be independent of original");
        assert_eq!(c.cycles, 999, "Original should reflect mutation");
    }

    #[test]
    fn hw_counters_delta_chaining() {
        let a = HwCounters { cycles: 300, instructions: 600, l1d_misses: 30, llc_misses: 10, branch_misses: 5 };
        let b = HwCounters { cycles: 200, instructions: 400, l1d_misses: 20, llc_misses: 5, branch_misses: 3 };
        let c = HwCounters { cycles: 50, instructions: 100, l1d_misses: 5, llc_misses: 1, branch_misses: 0 };
        let d1 = a.delta(&b);
        let d2 = d1.delta(&c);
        // d1 = (100, 200, 10, 5, 2), d2 = d1 - c = (50, 100, 5, 4, 2)
        assert_eq!(d2.cycles, 50);
        assert_eq!(d2.instructions, 100);
        assert_eq!(d2.l1d_misses, 5);
        assert_eq!(d2.llc_misses, 4);
        assert_eq!(d2.branch_misses, 2);
    }
}
