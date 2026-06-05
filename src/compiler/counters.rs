//! JIT compilation performance counters.
//!
//! Tracks statistics for the compilation pipeline phases:
//! - Phase 0: Scalar + SymExec (trace extraction)
//! - Phase 1: SemanticDAG (OpClass derivation)
//! - Phase 2: Fusion + HW constraints
//! - Phase 3: ISA Lowering (codegen)
//!
//! Counters are used for profiling and optimization of the compiler itself.

use std::fmt;

/// Performance counters for a single compilation phase.
#[derive(Debug, Clone, Default)]
pub struct PhaseCounters {
    /// Number of operations processed.
    pub ops_processed: u64,
    /// Number of fusion opportunities found.
    pub fusion_opportunities: u64,
    /// Number of fusions actually applied.
    pub fusions_applied: u64,
    /// Time spent in this phase (microseconds).
    pub time_us: u64,
}

impl PhaseCounters {
    /// Create a new zero-initialized counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment ops_processed by the given amount.
    pub fn inc_ops(&mut self, delta: u64) {
        self.ops_processed += delta;
    }

    /// Increment fusion_opportunities by 1.
    pub fn record_fusion_opportunity(&mut self) {
        self.fusion_opportunities += 1;
    }

    /// Increment fusions_applied by 1.
    pub fn record_fusion_applied(&mut self) {
        self.fusions_applied += 1;
    }

    /// Add time to the phase (microseconds).
    pub fn add_time(&mut self, us: u64) {
        self.time_us += us;
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.ops_processed = 0;
        self.fusion_opportunities = 0;
        self.fusions_applied = 0;
        self.time_us = 0;
    }

    /// Merge another counter into this one (sum all fields).
    pub fn merge(&mut self, other: &PhaseCounters) {
        self.ops_processed += other.ops_processed;
        self.fusion_opportunities += other.fusion_opportunities;
        self.fusions_applied += other.fusions_applied;
        self.time_us += other.time_us;
    }
}

impl fmt::Display for PhaseCounters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PhaseCounters(ops={}, fusions={}/{}, time={}us)",
            self.ops_processed,
            self.fusions_applied,
            self.fusion_opportunities,
            self.time_us
        )
    }
}

/// Aggregate counters for the entire compilation pipeline.
#[derive(Debug, Clone, Default)]
pub struct CompileCounters {
    /// Phase 0: Scalar + SymExec counters.
    pub phase0: PhaseCounters,
    /// Phase 1: SemanticDAG counters.
    pub phase1: PhaseCounters,
    /// Phase 2: Fusion + HW constraints counters.
    pub phase2: PhaseCounters,
    /// Phase 3: ISA Lowering counters.
    pub phase3: PhaseCounters,
    /// Total compiled layers.
    pub layers_compiled: u64,
    /// Total JIT code bytes generated.
    pub code_bytes: u64,
}

impl CompileCounters {
    /// Create a new zero-initialized counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a compiled layer.
    pub fn record_layer(&mut self, code_size: u64) {
        self.layers_compiled += 1;
        self.code_bytes += code_size;
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.phase0.reset();
        self.phase1.reset();
        self.phase2.reset();
        self.phase3.reset();
        self.layers_compiled = 0;
        self.code_bytes = 0;
    }

    /// Get total time across all phases (microseconds).
    pub fn total_time_us(&self) -> u64 {
        self.phase0.time_us
            + self.phase1.time_us
            + self.phase2.time_us
            + self.phase3.time_us
    }

    /// Get total fusions applied across all phases.
    pub fn total_fusions(&self) -> u64 {
        self.phase0.fusions_applied
            + self.phase1.fusions_applied
            + self.phase2.fusions_applied
            + self.phase3.fusions_applied
    }

    /// Merge another counter into this one.
    pub fn merge(&mut self, other: &CompileCounters) {
        self.phase0.merge(&other.phase0);
        self.phase1.merge(&other.phase1);
        self.phase2.merge(&other.phase2);
        self.phase3.merge(&other.phase3);
        self.layers_compiled += other.layers_compiled;
        self.code_bytes += other.code_bytes;
    }
}

impl fmt::Display for CompileCounters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompileCounters(layers={}, code={}KB, fusions={}, time={}us)",
            self.layers_compiled,
            self.code_bytes / 1024,
            self.total_fusions(),
            self.total_time_us()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PhaseCounters: Construction & Defaults ───────────────────────────────

    #[test]
    fn phase_counters_new_is_zero() {
        let c = PhaseCounters::new();
        assert_eq!(c.ops_processed, 0);
        assert_eq!(c.fusion_opportunities, 0);
        assert_eq!(c.fusions_applied, 0);
        assert_eq!(c.time_us, 0);
    }

    #[test]
    fn phase_counters_default_is_zero() {
        let c = PhaseCounters::default();
        assert_eq!(c.ops_processed, 0);
        assert_eq!(c.fusion_opportunities, 0);
        assert_eq!(c.fusions_applied, 0);
        assert_eq!(c.time_us, 0);
    }

    // ── PhaseCounters: Increment Operations ──────────────────────────────────

    #[test]
    fn phase_counters_inc_ops() {
        let mut c = PhaseCounters::new();
        c.inc_ops(10);
        assert_eq!(c.ops_processed, 10);
        c.inc_ops(5);
        assert_eq!(c.ops_processed, 15);
    }

    #[test]
    fn phase_counters_record_fusion_opportunity() {
        let mut c = PhaseCounters::new();
        c.record_fusion_opportunity();
        c.record_fusion_opportunity();
        c.record_fusion_opportunity();
        assert_eq!(c.fusion_opportunities, 3);
    }

    #[test]
    fn phase_counters_record_fusion_applied() {
        let mut c = PhaseCounters::new();
        c.record_fusion_applied();
        c.record_fusion_applied();
        assert_eq!(c.fusions_applied, 2);
    }

    #[test]
    fn phase_counters_add_time() {
        let mut c = PhaseCounters::new();
        c.add_time(100);
        c.add_time(50);
        assert_eq!(c.time_us, 150);
    }

    // ── PhaseCounters: Reset ─────────────────────────────────────────────────

    #[test]
    fn phase_counters_reset() {
        let mut c = PhaseCounters::new();
        c.inc_ops(100);
        c.record_fusion_opportunity();
        c.record_fusion_applied();
        c.add_time(1000);
        c.reset();
        assert_eq!(c.ops_processed, 0);
        assert_eq!(c.fusion_opportunities, 0);
        assert_eq!(c.fusions_applied, 0);
        assert_eq!(c.time_us, 0);
    }

    // ── PhaseCounters: Merge ─────────────────────────────────────────────────

    #[test]
    fn phase_counters_merge() {
        let mut c1 = PhaseCounters::new();
        c1.inc_ops(10);
        c1.record_fusion_opportunity();
        c1.record_fusion_applied();
        c1.add_time(100);

        let mut c2 = PhaseCounters::new();
        c2.inc_ops(5);
        c2.record_fusion_opportunity();
        c2.add_time(50);

        c1.merge(&c2);
        assert_eq!(c1.ops_processed, 15);
        assert_eq!(c1.fusion_opportunities, 2);
        assert_eq!(c1.fusions_applied, 1);
        assert_eq!(c1.time_us, 150);
    }

    // ── PhaseCounters: Display ───────────────────────────────────────────────

    #[test]
    fn phase_counters_display() {
        let mut c = PhaseCounters::new();
        c.inc_ops(42);
        c.record_fusion_opportunity();
        c.record_fusion_applied();
        c.add_time(1234);
        let s = format!("{}", c);
        assert!(s.contains("ops=42"));
        assert!(s.contains("fusions=1/1"));
        assert!(s.contains("time=1234us"));
    }

    // ── PhaseCounters: Edge Cases ────────────────────────────────────────────

    #[test]
    fn phase_counters_zero_merge() {
        let mut c = PhaseCounters::new();
        c.inc_ops(100);
        let zero = PhaseCounters::new();
        c.merge(&zero);
        assert_eq!(c.ops_processed, 100);
    }

    #[test]
    fn phase_counters_large_values() {
        let mut c = PhaseCounters::new();
        c.inc_ops(u64::MAX / 2);
        c.inc_ops(u64::MAX / 2 + 1);
        assert_eq!(c.ops_processed, u64::MAX);
    }

    // ── CompileCounters: Construction & Defaults ─────────────────────────────

    #[test]
    fn compile_counters_new_is_zero() {
        let c = CompileCounters::new();
        assert_eq!(c.layers_compiled, 0);
        assert_eq!(c.code_bytes, 0);
        assert_eq!(c.phase0.ops_processed, 0);
        assert_eq!(c.phase1.ops_processed, 0);
        assert_eq!(c.phase2.ops_processed, 0);
        assert_eq!(c.phase3.ops_processed, 0);
    }

    #[test]
    fn compile_counters_default_is_zero() {
        let c = CompileCounters::default();
        assert_eq!(c.layers_compiled, 0);
        assert_eq!(c.code_bytes, 0);
    }

    // ── CompileCounters: Record Layer ────────────────────────────────────────

    #[test]
    fn compile_counters_record_layer() {
        let mut c = CompileCounters::new();
        c.record_layer(1024);
        assert_eq!(c.layers_compiled, 1);
        assert_eq!(c.code_bytes, 1024);

        c.record_layer(2048);
        assert_eq!(c.layers_compiled, 2);
        assert_eq!(c.code_bytes, 3072);
    }

    // ── CompileCounters: Reset ───────────────────────────────────────────────

    #[test]
    fn compile_counters_reset() {
        let mut c = CompileCounters::new();
        c.record_layer(4096);
        c.phase0.inc_ops(100);
        c.phase1.inc_ops(200);
        c.phase2.inc_ops(300);
        c.phase3.inc_ops(400);
        c.reset();
        assert_eq!(c.layers_compiled, 0);
        assert_eq!(c.code_bytes, 0);
        assert_eq!(c.phase0.ops_processed, 0);
        assert_eq!(c.phase1.ops_processed, 0);
        assert_eq!(c.phase2.ops_processed, 0);
        assert_eq!(c.phase3.ops_processed, 0);
    }

    // ── CompileCounters: Total Time ──────────────────────────────────────────

    #[test]
    fn compile_counters_total_time_us() {
        let mut c = CompileCounters::new();
        c.phase0.add_time(100);
        c.phase1.add_time(200);
        c.phase2.add_time(300);
        c.phase3.add_time(400);
        assert_eq!(c.total_time_us(), 1000);
    }

    #[test]
    fn compile_counters_total_time_zero() {
        let c = CompileCounters::new();
        assert_eq!(c.total_time_us(), 0);
    }

    // ── CompileCounters: Total Fusions ───────────────────────────────────────

    #[test]
    fn compile_counters_total_fusions() {
        let mut c = CompileCounters::new();
        c.phase0.record_fusion_applied();
        c.phase0.record_fusion_applied();
        c.phase2.record_fusion_applied();
        c.phase3.record_fusion_applied();
        c.phase3.record_fusion_applied();
        c.phase3.record_fusion_applied();
        assert_eq!(c.total_fusions(), 6);
    }

    #[test]
    fn compile_counters_total_fusions_zero() {
        let c = CompileCounters::new();
        assert_eq!(c.total_fusions(), 0);
    }

    // ── CompileCounters: Merge ───────────────────────────────────────────────

    #[test]
    fn compile_counters_merge() {
        let mut c1 = CompileCounters::new();
        c1.record_layer(1024);
        c1.phase0.inc_ops(10);
        c1.phase2.record_fusion_applied();

        let mut c2 = CompileCounters::new();
        c2.record_layer(2048);
        c2.phase1.inc_ops(20);
        c2.phase3.record_fusion_applied();

        c1.merge(&c2);
        assert_eq!(c1.layers_compiled, 2);
        assert_eq!(c1.code_bytes, 3072);
        assert_eq!(c1.phase0.ops_processed, 10);
        assert_eq!(c1.phase1.ops_processed, 20);
        assert_eq!(c1.total_fusions(), 2);
    }

    // ── CompileCounters: Display ─────────────────────────────────────────────

    #[test]
    fn compile_counters_display() {
        let mut c = CompileCounters::new();
        c.record_layer(4096);
        c.phase2.record_fusion_applied();
        c.phase0.add_time(100);
        let s = format!("{}", c);
        assert!(s.contains("layers=1"));
        assert!(s.contains("code=4KB"));
        assert!(s.contains("fusions=1"));
        assert!(s.contains("time=100us"));
    }

    // ── CompileCounters: Edge Cases ──────────────────────────────────────────

    #[test]
    fn compile_counters_zero_merge() {
        let mut c = CompileCounters::new();
        c.record_layer(1024);
        let zero = CompileCounters::new();
        c.merge(&zero);
        assert_eq!(c.layers_compiled, 1);
        assert_eq!(c.code_bytes, 1024);
    }

    #[test]
    fn compile_counters_large_code_bytes() {
        let mut c = CompileCounters::new();
        c.record_layer(u64::MAX / 2);
        c.record_layer(u64::MAX / 2 + 1);
        assert_eq!(c.code_bytes, u64::MAX);
    }

    #[test]
    fn compile_counters_display_large_values() {
        let mut c = CompileCounters::new();
        c.record_layer(1024 * 1024 * 100); // 100 MB
        c.phase2.record_fusion_applied();
        c.phase2.record_fusion_applied();
        let s = format!("{}", c);
        assert!(s.contains("code=102400KB")); // 100 * 1024 KB
        assert!(s.contains("fusions=2"));
    }

    // ── PhaseCounters: Clone preserves values ────────────────────────────

    #[test]
    fn phase_counters_clone_preserves_values() {
        let mut c = PhaseCounters::new();
        c.inc_ops(42);
        c.record_fusion_opportunity();
        c.record_fusion_applied();
        c.add_time(999);

        let cloned = c.clone();
        assert_eq!(cloned.ops_processed, 42);
        assert_eq!(cloned.fusion_opportunities, 1);
        assert_eq!(cloned.fusions_applied, 1);
        assert_eq!(cloned.time_us, 999);
    }

    // ── CompileCounters: Clone preserves values ──────────────────────────

    #[test]
    fn compile_counters_clone_preserves_values() {
        let mut c = CompileCounters::new();
        c.record_layer(2048);
        c.phase0.inc_ops(10);
        c.phase2.record_fusion_applied();

        let cloned = c.clone();
        assert_eq!(cloned.layers_compiled, 1);
        assert_eq!(cloned.code_bytes, 2048);
        assert_eq!(cloned.phase0.ops_processed, 10);
        assert_eq!(cloned.phase2.fusions_applied, 1);
    }

    // ── PhaseCounters: Debug format ──────────────────────────────────────

    #[test]
    fn phase_counters_debug_format() {
        let mut c = PhaseCounters::new();
        c.inc_ops(5);
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("PhaseCounters"));
        assert!(debug_str.contains("ops_processed"));
    }

    // ── CompileCounters: Debug format ────────────────────────────────────

    #[test]
    fn compile_counters_debug_format() {
        let c = CompileCounters::new();
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("CompileCounters"));
        assert!(debug_str.contains("phase0"));
        assert!(debug_str.contains("phase3"));
    }

    // ── PhaseCounters: Self-merge doubles values ─────────────────────────

    #[test]
    fn phase_counters_merge_with_self() {
        let mut c = PhaseCounters::new();
        c.inc_ops(10);
        c.record_fusion_applied();
        c.add_time(100);

        let original = c.clone();
        c.merge(&original);

        assert_eq!(c.ops_processed, 20);
        assert_eq!(c.fusions_applied, 2);
        assert_eq!(c.time_us, 200);
    }

    // ── CompileCounters: Self-merge doubles values ───────────────────────

    #[test]
    fn compile_counters_merge_with_self() {
        let mut c = CompileCounters::new();
        c.record_layer(4096);
        c.phase1.inc_ops(7);

        let original = c.clone();
        c.merge(&original);

        assert_eq!(c.layers_compiled, 2);
        assert_eq!(c.code_bytes, 8192);
        assert_eq!(c.phase1.ops_processed, 14);
    }

    // ── PhaseCounters: inc_ops zero delta ────────────────────────────────

    #[test]
    fn phase_counters_inc_ops_zero_no_change() {
        let mut c = PhaseCounters::new();
        c.inc_ops(0);
        assert_eq!(c.ops_processed, 0);
    }

    // ── CompileCounters: total_time_single_phase ─────────────────────────

    #[test]
    fn compile_counters_total_time_single_phase() {
        let mut c = CompileCounters::new();
        c.phase3.add_time(500);
        assert_eq!(c.total_time_us(), 500);
        assert_eq!(c.phase0.time_us, 0);
        assert_eq!(c.phase1.time_us, 0);
        assert_eq!(c.phase2.time_us, 0);
    }

    // ── CompileCounters: total_fusions_single_phase ──────────────────────

    #[test]
    fn compile_counters_total_fusions_single_phase() {
        let mut c = CompileCounters::new();
        c.phase1.record_fusion_applied();
        c.phase1.record_fusion_applied();
        c.phase1.record_fusion_applied();
        assert_eq!(c.total_fusions(), 3);
        assert_eq!(c.phase0.fusions_applied, 0);
    }

    // ── PhaseCounters: Display zero state ────────────────────────────────

    #[test]
    fn phase_counters_display_zero_state() {
        let c = PhaseCounters::new();
        let s = format!("{}", c);
        assert!(s.contains("ops=0"));
        assert!(s.contains("fusions=0/0"));
        assert!(s.contains("time=0us"));
    }

    // ── CompileCounters: Display zero code bytes ─────────────────────────

    #[test]
    fn compile_counters_display_zero_code() {
        let c = CompileCounters::new();
        let s = format!("{}", c);
        assert!(s.contains("code=0KB"));
        assert!(s.contains("layers=0"));
        assert!(s.contains("fusions=0"));
    }
}
