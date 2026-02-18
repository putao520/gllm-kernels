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
#[derive(Debug, Clone, Copy, Default)]
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
}
