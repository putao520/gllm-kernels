//! JitContext — Unified hardware resource lifecycle tracker (SPEC 15)
//!
//! Tracks resource allocation/release during JIT ISA Lowering.
//! Provides budget gating, peak queries, leak detection, and diagnostic reports.
//!
//! ## Lifecycle
//! 1. `JitContext::new(isa_profile)` — created before ISA Lowering
//! 2. Lowerer calls `allocate`/`release` to track resource usage
//! 3. Any moment: `peak()`/`available()`/`snapshot()` for queries
//! 4. Dropped after ISA Lowering — zero runtime footprint

use std::collections::HashMap;

use crate::dispatch::device_profile::DeviceProfile;

use super::codegen::vm::isa_profile::{IsaProfile, Platform};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2.1 ResourceKind — Unified hardware resource types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Hardware resource kinds — unified across CPU/GPU/NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    // ── General-purpose registers ──
    /// GPR: x86 R0-R31(APX), ARM X0-X30, GPU R0-R255
    Gpr,
    /// SIMD/Vector: x86 YMM/ZMM, ARM V/Z, GPU VGPR, Apple SIMDgroup
    SimdVec,
    /// Mask/Predicate: x86 k0-k7, ARM SVE P0-P15, GPU %p0-%p7
    Predicate,

    // ── Accelerator tiles ──
    /// Tile: x86 AMX TMM0-7, ARM SME ZA, GPU TMEM
    Tile,
    /// Tile accumulator (logical): GPU WGMMA/tcgen05 fragment
    TileAccumulator,

    // ── Memory hierarchy ──
    /// CPU stack frame (spill slots + callee-save + ABI args)
    Stack,
    /// GPU shared memory: NVIDIA SMEM / AMD LDS / Metal Threadgroup
    SharedMem,
    /// GPU tensor memory: SM100+ TMEM (256KB/SM)
    TensorMem,

    // ── Synchronization primitives ──
    /// GPU mbarrier (SM90+) / CPU fence
    Barrier,
}

impl ResourceKind {
    /// All ResourceKind variants in canonical order.
    pub const ALL: [ResourceKind; 9] = [
        ResourceKind::Gpr,
        ResourceKind::SimdVec,
        ResourceKind::Predicate,
        ResourceKind::Tile,
        ResourceKind::TileAccumulator,
        ResourceKind::Stack,
        ResourceKind::SharedMem,
        ResourceKind::TensorMem,
        ResourceKind::Barrier,
    ];

    /// Whether this resource kind is available on the given platform.
    ///
    /// Based on JitResourceBudget::from_isa_profile semantics: a resource
    /// is "available" if `from_isa_profile` would produce a non-zero budget
    /// for it on the given platform.
    pub fn is_available_on(&self, platform: &Platform) -> bool {
        match self {
            Self::Gpr | Self::SimdVec => !platform.is_gpu(),
            Self::Stack => !platform.is_gpu(),
            Self::Predicate => matches!(platform,
                Platform::X86_64 { has_avx512: true, .. }
                | Platform::AArch64 { has_sve: true, .. }),
            Self::Tile => matches!(platform,
                Platform::X86_64 { has_amx: true, .. }
                | Platform::AArch64 { has_sme: true, .. }),
            Self::TileAccumulator => match platform {
                Platform::Cuda { sm_version, .. } => *sm_version >= 90,
                Platform::Hip { .. } => true,
                _ => false,
            },
            Self::SharedMem => matches!(platform,
                Platform::Cuda { .. } | Platform::Hip { .. } | Platform::Metal { .. }),
            Self::TensorMem => matches!(platform, Platform::Cuda { has_tmem: true, .. }),
            Self::Barrier => matches!(platform,
                Platform::Cuda { has_warp_spec: true, .. }),
        }
    }

    /// Whether this resource kind is tracked in bytes (memory-type).
    pub fn is_memory(&self) -> bool {
        matches!(self, ResourceKind::Stack | ResourceKind::SharedMem | ResourceKind::TensorMem)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2.3 ResourceState — Per-instance lifecycle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Lifecycle state of a single resource instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceState {
    /// Available for allocation.
    Free,
    /// Currently in use.
    Live {
        purpose: &'static str,
        alloc_instr: usize,
    },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 ResourceEvent — Timeline tracking
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A resource lifecycle event in the compilation timeline.
#[derive(Debug, Clone)]
pub struct ResourceEvent {
    pub instr_idx: usize,
    pub kind: ResourceKind,
    pub instance: usize,
    pub event_type: ResourceEventType,
    pub purpose: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceEventType {
    Allocate,
    Release,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 ResourceBudget — Device resource limits (full version)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Full device resource budget — 9 ResourceKind × capacity.
///
/// Derived from `IsaProfile` at JitContext creation time.
/// This is the SPEC 15 full version (vs the lightweight GEMM-only version
/// in `isa_hook.rs`).
#[derive(Debug, Clone)]
pub struct JitResourceBudget {
    pub gpr_total: usize,
    pub simd_vec_total: usize,
    pub predicate_total: usize,
    pub tile_total: usize,
    pub tile_accumulator_total: usize,
    pub stack_bytes: usize,
    pub shared_mem_bytes: usize,
    pub tensor_mem_bytes: usize,
    pub barrier_total: usize,
}

impl JitResourceBudget {
    /// Derive full resource budget from IsaProfile.
    pub fn from_isa_profile(profile: &IsaProfile) -> Self {
        let (shared_mem, tmem, barriers, tile_acc) = match &profile.platform {
            Platform::Cuda { shared_mem_kb, tmem_size_kb, has_warp_spec, sm_version, .. } => {
                let smem = shared_mem_kb * 1024;
                let tm = tmem_size_kb * 1024;
                let barriers = if *has_warp_spec { 2 } else { 0 };
                let acc = if *sm_version >= 90 { 4 } else { 0 };
                (smem, tm, barriers, acc)
            }
            Platform::Hip { lds_size_kb, .. } => {
                (lds_size_kb * 1024, 0, 0, 4) // MFMA accumulator
            }
            Platform::Metal { threadgroup_mem_kb, .. } => {
                (threadgroup_mem_kb * 1024, 0, 0, 0)
            }
            _ => (0, 0, 0, 0),
        };

        Self {
            gpr_total: profile.gpr_regs.len(),
            simd_vec_total: profile.vec_regs.len(),
            predicate_total: profile.mask_regs.len(),
            tile_total: profile.tile_regs.len(),
            tile_accumulator_total: tile_acc,
            stack_bytes: if profile.platform.is_gpu() { 0 } else { 4096 },
            shared_mem_bytes: shared_mem,
            tensor_mem_bytes: tmem,
            barrier_total: barriers,
        }
    }

    /// Get capacity for a specific resource kind.
    pub fn capacity(&self, kind: ResourceKind) -> usize {
        match kind {
            ResourceKind::Gpr => self.gpr_total,
            ResourceKind::SimdVec => self.simd_vec_total,
            ResourceKind::Predicate => self.predicate_total,
            ResourceKind::Tile => self.tile_total,
            ResourceKind::TileAccumulator => self.tile_accumulator_total,
            ResourceKind::Stack => self.stack_bytes,
            ResourceKind::SharedMem => self.shared_mem_bytes,
            ResourceKind::TensorMem => self.tensor_mem_bytes,
            ResourceKind::Barrier => self.barrier_total,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5.3 ResourceExhausted — Budget exceeded error
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Resource budget exceeded — contains diagnostics.
#[derive(Debug, Clone)]
pub struct ResourceExhausted {
    pub kind: ResourceKind,
    pub requested: usize,
    pub capacity: usize,
    pub current_live: usize,
    pub peak: usize,
    pub suggestion: &'static str,
}

impl std::fmt::Display for ResourceExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} exhausted: requested={}, capacity={}, live={}, peak={}. {}",
            self.kind, self.requested, self.capacity, self.current_live, self.peak, self.suggestion
        )
    }
}

impl std::error::Error for ResourceExhausted {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5.2 ResourceSnapshot — Time-point snapshot
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Resource usage at a specific point during compilation.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    pub instr_idx: usize,
    pub live: HashMap<ResourceKind, usize>,
    pub peak: HashMap<ResourceKind, usize>,
    pub available: HashMap<ResourceKind, usize>,
    pub mem_used: HashMap<ResourceKind, usize>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §9 ResourceReport — Post-compilation diagnostics
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Post-compilation resource usage report.
#[derive(Debug, Clone)]
pub struct ResourceReport {
    pub peak_usage: HashMap<ResourceKind, usize>,
    pub capacity: JitResourceBudget,
    pub utilization: HashMap<ResourceKind, f64>,
    pub warnings: Vec<ResourceWarning>,
}

#[derive(Debug, Clone)]
pub enum ResourceWarning {
    /// Resource utilization exceeds 90%.
    NearExhaustion { kind: ResourceKind, peak: usize, capacity: usize },
    /// Resource allocated but never released.
    Leak { kind: ResourceKind, instance: usize, purpose: &'static str },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §5 JitContext — Core structure
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// JIT compilation context — ISA Lowering hardware resource lifecycle tracker.
///
/// !Sync + !Send: single-threaded, compile-time only.
/// Zero runtime footprint: dropped after ISA Lowering.
pub struct JitContext {
    budget: JitResourceBudget,
    resources: HashMap<ResourceKind, Vec<ResourceState>>,
    events: Vec<ResourceEvent>,
    peak_live: HashMap<ResourceKind, usize>,
    stack_used: usize,
    smem_used: usize,
    tmem_used: usize,
    current_instr: usize,
}

impl JitContext {
    /// Create from IsaProfile.
    pub fn new(profile: &IsaProfile) -> Self {
        let budget = JitResourceBudget::from_isa_profile(profile);
        let mut resources = HashMap::new();
        for kind in ResourceKind::ALL {
            let cap = budget.capacity(kind);
            if cap > 0 {
                resources.insert(kind, vec![ResourceState::Free; cap]);
            }
        }
        let mut peak_live = HashMap::new();
        for kind in ResourceKind::ALL {
            peak_live.insert(kind, 0);
        }
        Self {
            budget,
            resources,
            events: Vec::new(),
            peak_live,
            stack_used: 0,
            smem_used: 0,
            tmem_used: 0,
            current_instr: 0,
        }
    }

    /// Create from DeviceProfile (convenience).
    pub fn from_device_profile(profile: &DeviceProfile) -> Self {
        let isa = IsaProfile::from_device_profile(profile);
        Self::new(&isa)
    }

    // ── Resource allocation ──────────────────────────────────────

    /// Allocate a resource instance. Returns instance index.
    pub fn allocate(
        &mut self,
        kind: ResourceKind,
        purpose: &'static str,
    ) -> Result<usize, ResourceExhausted> {
        let pool = self.resources.get_mut(&kind).ok_or(ResourceExhausted {
            kind,
            requested: 1,
            capacity: 0,
            current_live: 0,
            peak: 0,
            suggestion: "resource kind not available on this device",
        })?;

        for (i, state) in pool.iter_mut().enumerate() {
            if *state == ResourceState::Free {
                *state = ResourceState::Live {
                    purpose,
                    alloc_instr: self.current_instr,
                };
                let live = pool.iter().filter(|s| matches!(s, ResourceState::Live { .. })).count();
                if live > self.peak_live[&kind] {
                    self.peak_live.insert(kind, live);
                }
                self.events.push(ResourceEvent {
                    instr_idx: self.current_instr,
                    kind,
                    instance: i,
                    event_type: ResourceEventType::Allocate,
                    purpose,
                });
                self.debug_log_alloc(kind, i, purpose, live);
                return Ok(i);
            }
        }

        let live = pool.iter().filter(|s| matches!(s, ResourceState::Live { .. })).count();
        Err(ResourceExhausted {
            kind,
            requested: 1,
            capacity: pool.len(),
            current_live: live,
            peak: self.peak_live[&kind],
            suggestion: Self::suggestion_for(kind),
        })
    }

    /// Release a resource instance.
    ///
    /// Panics on double-free (release of already Free resource) per REQ-JCTX-021.
    pub fn release(&mut self, kind: ResourceKind, instance: usize) {
        let pool = self.resources.get_mut(&kind).unwrap_or_else(|| {
            panic!("release({:?}, {}): resource kind not tracked", kind, instance)
        });
        let state = &mut pool[instance];
        let purpose = match state {
            ResourceState::Free => {
                panic!(
                    "double-free detected: release({:?}, {}) on already Free resource (REQ-JCTX-021)",
                    kind, instance
                );
            }
            ResourceState::Live { purpose, .. } => *purpose,
        };
        self.events.push(ResourceEvent {
            instr_idx: self.current_instr,
            kind,
            instance,
            event_type: ResourceEventType::Release,
            purpose,
        });
        *state = ResourceState::Free;
        let live = pool.iter().filter(|s| matches!(s, ResourceState::Live { .. })).count();
        self.debug_log_release(kind, instance, live);
    }

    /// Pre-allocate N consecutive instances (for SMEM/TMEM regions).
    pub fn allocate_region(
        &mut self,
        kind: ResourceKind,
        count: usize,
        purpose: &'static str,
    ) -> Result<usize, ResourceExhausted> {
        let pool = self.resources.get_mut(&kind).ok_or(ResourceExhausted {
            kind,
            requested: count,
            capacity: 0,
            current_live: 0,
            peak: 0,
            suggestion: "resource kind not available on this device",
        })?;

        // Find first contiguous run of `count` Free slots
        let mut run_start = None;
        let mut run_len = 0;
        for (i, state) in pool.iter().enumerate() {
            if *state == ResourceState::Free {
                if run_start.is_none() {
                    run_start = Some(i);
                }
                run_len += 1;
                if run_len >= count {
                    break;
                }
            } else {
                run_start = None;
                run_len = 0;
            }
        }

        let start = run_start.ok_or_else(|| {
            let live = pool.iter().filter(|s| matches!(s, ResourceState::Live { .. })).count();
            ResourceExhausted {
                kind,
                requested: count,
                capacity: pool.len(),
                current_live: live,
                peak: self.peak_live[&kind],
                suggestion: Self::suggestion_for(kind),
            }
        })?;

        for i in start..start + count {
            pool[i] = ResourceState::Live {
                purpose,
                alloc_instr: self.current_instr,
            };
            self.events.push(ResourceEvent {
                instr_idx: self.current_instr,
                kind,
                instance: i,
                event_type: ResourceEventType::Allocate,
                purpose,
            });
        }

        let live = pool.iter().filter(|s| matches!(s, ResourceState::Live { .. })).count();
        if live > self.peak_live[&kind] {
            self.peak_live.insert(kind, live);
        }

        Ok(start)
    }

    // ── Query interface ──────────────────────────────────────────

    /// Current live instance count for a resource kind.
    pub fn live_count(&self, kind: ResourceKind) -> usize {
        self.resources
            .get(&kind)
            .map(|pool| pool.iter().filter(|s| matches!(s, ResourceState::Live { .. })).count())
            .unwrap_or(0)
    }

    /// Historical peak live count.
    pub fn peak(&self, kind: ResourceKind) -> usize {
        self.peak_live.get(&kind).copied().unwrap_or(0)
    }

    /// Available instance count.
    pub fn available(&self, kind: ResourceKind) -> usize {
        self.capacity(kind).saturating_sub(self.live_count(kind))
    }

    /// Physical capacity.
    pub fn capacity(&self, kind: ResourceKind) -> usize {
        self.budget.capacity(kind)
    }

    /// Memory-type resource used bytes.
    pub fn mem_used(&self, kind: ResourceKind) -> usize {
        match kind {
            ResourceKind::Stack => self.stack_used,
            ResourceKind::SharedMem => self.smem_used,
            ResourceKind::TensorMem => self.tmem_used,
            _ => 0,
        }
    }

    /// Memory-type resource available bytes.
    pub fn mem_available(&self, kind: ResourceKind) -> usize {
        self.capacity(kind).saturating_sub(self.mem_used(kind))
    }

    // ── Snapshot ─────────────────────────────────────────────────

    /// Take a resource usage snapshot at the current point.
    pub fn snapshot(&self) -> ResourceSnapshot {
        let mut live = HashMap::new();
        let mut available = HashMap::new();
        let mut mem_used = HashMap::new();
        for &kind in &ResourceKind::ALL {
            live.insert(kind, self.live_count(kind));
            available.insert(kind, self.available(kind));
            mem_used.insert(kind, self.mem_used(kind));
        }
        ResourceSnapshot {
            instr_idx: self.current_instr,
            live,
            peak: self.peak_live.clone(),
            available,
            mem_used,
        }
    }

    // ── Instruction advance ──────────────────────────────────────

    /// Advance the instruction pointer.
    pub fn advance_to(&mut self, instr_idx: usize) {
        assert!(
            instr_idx >= self.current_instr,
            "advance_to: instr_idx must be monotonic (current={}, got={})",
            self.current_instr,
            instr_idx
        );
        self.current_instr = instr_idx;
    }

    // ── Dynamic budget updates ───────────────────────────────────

    /// Update stack budget (after StackFrame computation).
    pub fn update_stack_budget(&mut self, bytes: usize) {
        self.budget.stack_bytes = bytes;
    }

    /// Declare SMEM usage (after shared memory allocation).
    pub fn declare_smem_usage(&mut self, bytes: usize) {
        self.smem_used += bytes;
    }

    /// Declare TMEM usage (after tcgen05.alloc).
    pub fn declare_tmem_usage(&mut self, bytes: usize) {
        self.tmem_used += bytes;
    }

    // ── Diagnostics ──────────────────────────────────────────────

    /// Generate post-compilation resource report.
    pub fn usage_report(&self) -> ResourceReport {
        let mut utilization = HashMap::new();
        let mut warnings = Vec::new();

        for &kind in &ResourceKind::ALL {
            let peak = self.peak(kind);
            let cap = self.capacity(kind);
            let util = if cap > 0 { peak as f64 / cap as f64 } else { 0.0 };
            utilization.insert(kind, util);

            if cap > 0 && util > 0.9 {
                warnings.push(ResourceWarning::NearExhaustion {
                    kind,
                    peak,
                    capacity: cap,
                });
            }

            // Leak detection (REQ-JCTX-022)
            if let Some(pool) = self.resources.get(&kind) {
                for (i, state) in pool.iter().enumerate() {
                    if let ResourceState::Live { purpose, .. } = state {
                        // Tile/Barrier/SMEM/TMEM are hardware-managed, skip
                        if !matches!(
                            kind,
                            ResourceKind::Tile | ResourceKind::Barrier | ResourceKind::SharedMem | ResourceKind::TensorMem
                        ) {
                            warnings.push(ResourceWarning::Leak {
                                kind,
                                instance: i,
                                purpose,
                            });
                        }
                    }
                }
            }
        }

        ResourceReport {
            peak_usage: self.peak_live.clone(),
            capacity: self.budget.clone(),
            utilization,
            warnings,
        }
    }

    /// Borrow the budget (for RegAllocator integration).
    pub fn budget(&self) -> &JitResourceBudget {
        &self.budget
    }

    /// Total event count in the timeline.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    // ── Debug logging ────────────────────────────────────────────

    fn debug_log_alloc(&self, kind: ResourceKind, instance: usize, purpose: &'static str, live: usize) {
        if std::env::var("GLLM_DEBUG_RESOURCE").as_deref() == Ok("1") {
            eprintln!(
                "[JitContext] instr={} allocate {:?}:{} {:?} (live={}, peak={}, cap={})",
                self.current_instr,
                kind,
                instance,
                purpose,
                live,
                self.peak(kind),
                self.capacity(kind),
            );
        }
    }

    fn debug_log_release(&self, kind: ResourceKind, instance: usize, live: usize) {
        if std::env::var("GLLM_DEBUG_RESOURCE").as_deref() == Ok("1") {
            eprintln!(
                "[JitContext] instr={} release {:?}:{} (live={}, peak={}, cap={})",
                self.current_instr,
                kind,
                instance,
                live,
                self.peak(kind),
                self.capacity(kind),
            );
        }
    }

    fn suggestion_for(kind: ResourceKind) -> &'static str {
        match kind {
            ResourceKind::Gpr => "reduce register pressure or spill to stack",
            ResourceKind::SimdVec => "reduce vector register pressure or use narrower types",
            ResourceKind::Predicate => "reduce mask register usage",
            ResourceKind::Tile => "reduce tile count or release unused tiles",
            ResourceKind::TileAccumulator => "reduce WGMMA fragment count",
            ResourceKind::Stack => "reduce stack usage or increase stack budget",
            ResourceKind::SharedMem => "reduce SMEM tile size or epilogue chain length",
            ResourceKind::TensorMem => "reduce TMEM allocation size",
            ResourceKind::Barrier => "reduce synchronization points",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile() -> IsaProfile {
        IsaProfile::from_device_profile(&DeviceProfile::detect())
    }

    #[test]
    fn jit_context_new_populates_resources() {
        let profile = make_profile();
        let ctx = JitContext::new(&profile);
        // GPR and SimdVec should always be populated on any CPU
        assert!(ctx.capacity(ResourceKind::Gpr) > 0);
        assert!(ctx.capacity(ResourceKind::SimdVec) > 0);
        // All peak values start at 0
        for &kind in &ResourceKind::ALL {
            assert_eq!(ctx.peak(kind), 0);
        }
    }

    #[test]
    fn allocate_release_lifecycle() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let idx = ctx.allocate(ResourceKind::Gpr, "test_counter").unwrap();
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 1);
        assert_eq!(ctx.peak(ResourceKind::Gpr), 1);
        assert_eq!(ctx.available(ResourceKind::Gpr), ctx.capacity(ResourceKind::Gpr) - 1);

        ctx.release(ResourceKind::Gpr, idx);
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 0);
        assert_eq!(ctx.peak(ResourceKind::Gpr), 1); // peak stays
        assert_eq!(ctx.available(ResourceKind::Gpr), ctx.capacity(ResourceKind::Gpr));
    }

    #[test]
    fn peak_tracks_maximum() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        let mut allocated = Vec::new();

        for i in 0..cap.min(3) {
            let idx = ctx.allocate(ResourceKind::Gpr, "peak_test").unwrap();
            allocated.push(idx);
            assert_eq!(ctx.peak(ResourceKind::Gpr), i + 1);
        }

        for &idx in &allocated {
            ctx.release(ResourceKind::Gpr, idx);
        }
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 0);
        assert_eq!(ctx.peak(ResourceKind::Gpr), cap.min(3));
    }

    #[test]
    fn budget_exceeded_returns_error() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);

        // Exhaust all GPRs
        for i in 0..cap {
            ctx.allocate(ResourceKind::Gpr, "exhaust_test").unwrap();
        }

        // One more should fail
        let err = ctx.allocate(ResourceKind::Gpr, "overflow").unwrap_err();
        assert_eq!(err.kind, ResourceKind::Gpr);
        assert_eq!(err.current_live, cap);
    }

    #[test]
    fn double_free_panics() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let idx = ctx.allocate(ResourceKind::Gpr, "df_test").unwrap();
        ctx.release(ResourceKind::Gpr, idx);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ctx = ctx;
            ctx.release(ResourceKind::Gpr, idx);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn advance_to_monotonic() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.advance_to(10);
        ctx.advance_to(20);
        assert_eq!(ctx.current_instr, 20);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ctx = ctx;
            ctx.advance_to(15); // non-monotonic
        }));
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_captures_state() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.allocate(ResourceKind::Gpr, "snap_test").unwrap();
        ctx.advance_to(42);

        let snap = ctx.snapshot();
        assert_eq!(snap.instr_idx, 42);
        assert_eq!(snap.live[&ResourceKind::Gpr], 1);
        assert_eq!(snap.peak[&ResourceKind::Gpr], 1);
        assert_eq!(snap.available[&ResourceKind::Gpr], ctx.capacity(ResourceKind::Gpr) - 1);
    }

    #[test]
    fn memory_tracking() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        ctx.update_stack_budget(8192);
        assert_eq!(ctx.capacity(ResourceKind::Stack), 8192);

        ctx.declare_smem_usage(4096);
        assert_eq!(ctx.mem_used(ResourceKind::SharedMem), 4096);

        ctx.declare_tmem_usage(2048);
        assert_eq!(ctx.mem_used(ResourceKind::TensorMem), 2048);
    }

    #[test]
    fn usage_report_no_leaks() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let idx = ctx.allocate(ResourceKind::Gpr, "report_test").unwrap();
        ctx.release(ResourceKind::Gpr, idx);

        let report = ctx.usage_report();
        assert!(report.peak_usage[&ResourceKind::Gpr] >= 1);
        // No leaks because we released
        let leaks: Vec<_> = report.warnings.iter().filter(|w| matches!(w, ResourceWarning::Leak { .. })).collect();
        assert!(leaks.is_empty());
    }

    #[test]
    fn usage_report_detects_leak() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.allocate(ResourceKind::Gpr, "leaked").unwrap();

        let report = ctx.usage_report();
        let leaks: Vec<_> = report.warnings.iter().filter(|w| matches!(w, ResourceWarning::Leak { .. })).collect();
        assert!(!leaks.is_empty());
    }

    #[test]
    fn allocate_region_contiguous() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 4 {
            return; // skip on very constrained configs
        }

        let start = ctx.allocate_region(ResourceKind::Gpr, 3, "region_test").unwrap();
        assert!(start + 3 <= cap);
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 3);
    }

    #[test]
    fn event_timeline_tracks_allocations() {
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        assert_eq!(ctx.event_count(), 0);

        ctx.advance_to(10);
        let idx = ctx.allocate(ResourceKind::Gpr, "ev_test").unwrap();
        assert_eq!(ctx.event_count(), 1);
        assert_eq!(ctx.events[0].instr_idx, 10);
        assert_eq!(ctx.events[0].event_type, ResourceEventType::Allocate);
        assert_eq!(ctx.events[0].instance, idx);

        ctx.advance_to(20);
        ctx.release(ResourceKind::Gpr, idx);
        assert_eq!(ctx.event_count(), 2);
        assert_eq!(ctx.events[1].instr_idx, 20);
        assert_eq!(ctx.events[1].event_type, ResourceEventType::Release);
    }

    #[test]
    fn resource_kind_all_count() {
        assert_eq!(ResourceKind::ALL.len(), 9);
    }

    #[test]
    fn memory_kinds_correct() {
        assert!(ResourceKind::Stack.is_memory());
        assert!(ResourceKind::SharedMem.is_memory());
        assert!(ResourceKind::TensorMem.is_memory());
        assert!(!ResourceKind::Gpr.is_memory());
        assert!(!ResourceKind::Tile.is_memory());
    }

    // ── New tests (13) ─────────────────────────────────────────────

    #[test]
    fn allocate_region_fragmented_pool() {
        // Arrange: allocate 3, release middle one, then request 2 contiguous
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 5 {
            return;
        }

        // Act
        let i0 = ctx.allocate(ResourceKind::Gpr, "a").unwrap();
        let i1 = ctx.allocate(ResourceKind::Gpr, "b").unwrap();
        let i2 = ctx.allocate(ResourceKind::Gpr, "c").unwrap();
        ctx.release(ResourceKind::Gpr, i1); // create a gap at i1

        // Assert: allocate_region should skip the gap and find a contiguous run
        let start = ctx.allocate_region(ResourceKind::Gpr, 2, "frag_test").unwrap();
        // The region of 2 contiguous slots must be after the gap or wrap around;
        // it must NOT include the freed slot as part of a 2-run starting there
        // unless i2+1 and i2+2 are free.
        assert!(start + 2 <= cap);
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 4); // i0, i2, +2 from region
    }

    #[test]
    fn allocate_untracked_resource_kind_fails() {
        // Arrange: on CPU, GPU-only resources (SharedMem/TensorMem/Barrier) have 0 capacity
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        // Act
        let result = ctx.allocate(ResourceKind::SharedMem, "gpu_only");

        // Assert: should fail because CPU has no shared memory
        if ctx.capacity(ResourceKind::SharedMem) == 0 {
            let err = result.unwrap_err();
            assert_eq!(err.kind, ResourceKind::SharedMem);
            assert_eq!(err.capacity, 0);
            assert_eq!(err.suggestion, "resource kind not available on this device");
        }
    }

    #[test]
    fn mem_available_calculation() {
        // Arrange
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.update_stack_budget(8192);

        // Act: declare some stack usage
        ctx.declare_smem_usage(1024);

        // Assert: Stack mem_available should be 8192 - 0 = 8192 (no stack_used tracking via declare)
        assert_eq!(ctx.mem_available(ResourceKind::Stack), 8192);
        // SharedMem: capacity - declared usage
        if ctx.capacity(ResourceKind::SharedMem) > 0 {
            assert_eq!(
                ctx.mem_available(ResourceKind::SharedMem),
                ctx.capacity(ResourceKind::SharedMem) - 1024
            );
        }
    }

    #[test]
    fn cumulative_memory_declarations() {
        // Arrange: multiple declare calls should accumulate
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        // Act
        ctx.declare_smem_usage(1024);
        ctx.declare_smem_usage(2048);
        ctx.declare_tmem_usage(512);
        ctx.declare_tmem_usage(256);

        // Assert
        assert_eq!(ctx.mem_used(ResourceKind::SharedMem), 3072); // 1024 + 2048
        assert_eq!(ctx.mem_used(ResourceKind::TensorMem), 768);  // 512 + 256
    }

    #[test]
    fn usage_report_near_exhaustion_warning() {
        // Arrange: allocate enough GPRs to exceed 90% utilization
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 3 {
            return;
        }
        // Allocate all but one to get >90% utilization when cap is small,
        // or allocate all to guarantee 100%
        let mut allocated = Vec::new();
        for _ in 0..cap {
            allocated.push(ctx.allocate(ResourceKind::Gpr, "stress").unwrap());
        }

        // Act
        let report = ctx.usage_report();

        // Assert
        let near: Vec<_> = report.warnings.iter()
            .filter(|w| matches!(w, ResourceWarning::NearExhaustion { kind: ResourceKind::Gpr, .. }))
            .collect();
        assert_eq!(near.len(), 1);
        if let ResourceWarning::NearExhaustion { peak, capacity, .. } = near[0] {
            assert_eq!(*peak, cap);
            assert_eq!(*capacity, cap);
        }

        // Cleanup
        for idx in allocated {
            ctx.release(ResourceKind::Gpr, idx);
        }
    }

    #[test]
    fn resource_exhausted_error_fields() {
        // Arrange: exhaust all SimdVec registers
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::SimdVec);

        for _ in 0..cap {
            ctx.allocate(ResourceKind::SimdVec, "fill").unwrap();
        }

        // Act
        let err = ctx.allocate(ResourceKind::SimdVec, "overflow").unwrap_err();

        // Assert: verify all fields of ResourceExhausted
        assert_eq!(err.kind, ResourceKind::SimdVec);
        assert_eq!(err.requested, 1);
        assert_eq!(err.capacity, cap);
        assert_eq!(err.current_live, cap);
        assert_eq!(err.peak, cap);
        // Suggestion should be SIMD-specific
        assert_eq!(err.suggestion, "reduce vector register pressure or use narrower types");
    }

    #[test]
    fn snapshot_includes_memory_usage() {
        // Arrange
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.update_stack_budget(4096);
        ctx.declare_smem_usage(2048);

        // Act
        let snap = ctx.snapshot();

        // Assert: memory fields should be populated for all kinds
        assert_eq!(snap.mem_used[&ResourceKind::SharedMem], 2048);
        assert_eq!(snap.mem_used[&ResourceKind::Stack], 0); // no stack declared via declare
        assert_eq!(snap.mem_used[&ResourceKind::TensorMem], 0);
        // Non-memory kinds should have 0
        assert_eq!(snap.mem_used[&ResourceKind::Gpr], 0);
    }

    #[test]
    fn allocate_region_exhausted_no_contiguous_run() {
        // Arrange: create a checkerboard pattern so no 2-contiguous run exists
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 4 {
            return;
        }

        // Allocate all slots
        let mut all = Vec::new();
        for _ in 0..cap {
            all.push(ctx.allocate(ResourceKind::Gpr, "chess").unwrap());
        }
        // Release every other slot to create fragmentation (no 2-contiguous free run if cap is even)
        for i in (0..cap).step_by(2) {
            ctx.release(ResourceKind::Gpr, all[i]);
        }

        // Act: if cap is even, no 2-contiguous free slots exist
        if cap % 2 == 0 {
            let result = ctx.allocate_region(ResourceKind::Gpr, 2, "no_room");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind, ResourceKind::Gpr);
        }
    }

    #[test]
    fn region_allocation_generates_events() {
        // Arrange
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 3 {
            return;
        }
        ctx.advance_to(100);
        let events_before = ctx.event_count();

        // Act
        let start = ctx.allocate_region(ResourceKind::Gpr, 3, "evt_region").unwrap();

        // Assert: should generate 3 allocation events
        assert_eq!(ctx.event_count(), events_before + 3);
        for i in 0..3 {
            let evt = &ctx.events[events_before + i];
            assert_eq!(evt.instr_idx, 100);
            assert_eq!(evt.kind, ResourceKind::Gpr);
            assert_eq!(evt.instance, start + i);
            assert_eq!(evt.event_type, ResourceEventType::Allocate);
            assert_eq!(evt.purpose, "evt_region");
        }
    }

    #[test]
    fn from_device_profile_creates_context() {
        // Arrange & Act
        let ctx = JitContext::from_device_profile(&DeviceProfile::detect());

        // Assert: same invariants as direct construction
        assert!(ctx.capacity(ResourceKind::Gpr) > 0);
        assert!(ctx.capacity(ResourceKind::SimdVec) > 0);
        for &kind in &ResourceKind::ALL {
            assert_eq!(ctx.peak(kind), 0);
            assert_eq!(ctx.live_count(kind), 0);
        }
    }

    #[test]
    fn peak_preserved_after_full_release() {
        // Arrange: allocate different amounts, verify peak sticks
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::SimdVec);
        if cap < 2 {
            return;
        }

        // Act: allocate 2, release both, allocate 1
        let v0 = ctx.allocate(ResourceKind::SimdVec, "p1").unwrap();
        let v1 = ctx.allocate(ResourceKind::SimdVec, "p2").unwrap();
        assert_eq!(ctx.peak(ResourceKind::SimdVec), 2);

        ctx.release(ResourceKind::SimdVec, v0);
        ctx.release(ResourceKind::SimdVec, v1);
        assert_eq!(ctx.peak(ResourceKind::SimdVec), 2); // peak stays after release

        let _v2 = ctx.allocate(ResourceKind::SimdVec, "p3").unwrap();

        // Assert
        assert_eq!(ctx.live_count(ResourceKind::SimdVec), 1);
        assert_eq!(ctx.peak(ResourceKind::SimdVec), 2); // peak unchanged (1 < 2)
    }

    #[test]
    fn resource_exhausted_display_format() {
        // Arrange
        let err = ResourceExhausted {
            kind: ResourceKind::Tile,
            requested: 2,
            capacity: 4,
            current_live: 4,
            peak: 4,
            suggestion: "reduce tile count or release unused tiles",
        };

        // Act
        let msg = format!("{}", err);

        // Assert: should contain all key diagnostic info
        assert!(msg.contains("Tile"));
        assert!(msg.contains("requested=2"));
        assert!(msg.contains("capacity=4"));
        assert!(msg.contains("live=4"));
        assert!(msg.contains("peak=4"));
        assert!(msg.contains("reduce tile count"));
    }

    #[test]
    fn update_stack_budget_adjusts_capacity() {
        // Arrange
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let initial = ctx.capacity(ResourceKind::Stack);

        // Act: increase stack budget
        ctx.update_stack_budget(initial + 8192);

        // Assert
        assert_eq!(ctx.capacity(ResourceKind::Stack), initial + 8192);
        // Budget accessor should reflect the change
        assert_eq!(ctx.budget().stack_bytes, initial + 8192);
    }

    // ── New tests (wave-12kfe, +10) ────────────────────────────────

    #[test]
    fn allocate_region_untracked_kind_fails() {
        // Arrange: on CPU, SharedMem has 0 capacity so allocate_region should fail
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        if ctx.capacity(ResourceKind::SharedMem) == 0 {
            // Act
            let err = ctx.allocate_region(ResourceKind::SharedMem, 2, "gpu_region").unwrap_err();

            // Assert
            assert_eq!(err.kind, ResourceKind::SharedMem);
            assert_eq!(err.requested, 2);
            assert_eq!(err.capacity, 0);
            assert_eq!(err.suggestion, "resource kind not available on this device");
        }
    }

    #[test]
    fn usage_report_utilization_calculation() {
        // Arrange: allocate exactly half the GPRs, verify utilization is ~50%
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 2 {
            return;
        }
        let half = cap / 2;
        let mut allocated = Vec::new();
        for _ in 0..half {
            allocated.push(ctx.allocate(ResourceKind::Gpr, "util_test").unwrap());
        }

        // Act
        let report = ctx.usage_report();

        // Assert
        let util = report.utilization[&ResourceKind::Gpr];
        let expected = half as f64 / cap as f64;
        assert!((util - expected).abs() < 0.01, "expected {}, got {}", expected, util);

        // Cleanup
        for idx in allocated {
            ctx.release(ResourceKind::Gpr, idx);
        }
    }

    #[test]
    fn live_count_untracked_kind_returns_zero() {
        // Arrange: Predicate might be 0 on some CPU-only profiles
        let profile = make_profile();
        let ctx = JitContext::new(&profile);

        // Act & Assert: any kind with 0 capacity should report 0 live
        for &kind in &ResourceKind::ALL {
            if ctx.capacity(kind) == 0 {
                assert_eq!(ctx.live_count(kind), 0);
                assert_eq!(ctx.peak(kind), 0);
                assert_eq!(ctx.available(kind), 0);
            }
        }
    }

    #[test]
    fn mem_used_non_memory_kinds_zero() {
        // Arrange
        let profile = make_profile();
        let ctx = JitContext::new(&profile);

        // Act & Assert: non-memory kinds always return 0 from mem_used
        let non_memory_kinds = [
            ResourceKind::Gpr,
            ResourceKind::SimdVec,
            ResourceKind::Predicate,
            ResourceKind::Tile,
            ResourceKind::TileAccumulator,
            ResourceKind::Barrier,
        ];
        for &kind in &non_memory_kinds {
            assert_eq!(ctx.mem_used(kind), 0, "{:?} should have 0 mem_used", kind);
        }
    }

    #[test]
    fn mem_available_non_memory_kinds_equals_zero() {
        // Arrange: mem_available for non-memory kinds = capacity.saturating_sub(0) = capacity
        // But since mem_used returns 0 for non-memory, mem_available = capacity
        let profile = make_profile();
        let ctx = JitContext::new(&profile);

        // Act & Assert: mem_available for non-memory kinds equals their full capacity
        // (because mem_used is always 0 for them)
        let non_memory_kinds = [
            ResourceKind::Gpr,
            ResourceKind::SimdVec,
            ResourceKind::Predicate,
            ResourceKind::Tile,
            ResourceKind::TileAccumulator,
            ResourceKind::Barrier,
        ];
        for &kind in &non_memory_kinds {
            assert_eq!(
                ctx.mem_available(kind),
                ctx.capacity(kind),
                "{:?} mem_available should equal capacity when mem_used=0",
                kind
            );
        }
    }

    #[test]
    fn resource_state_live_records_purpose_and_instr() {
        // Arrange: verify that ResourceState::Live captures purpose and alloc_instr
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.advance_to(77);

        // Act
        let idx = ctx.allocate(ResourceKind::Gpr, "purpose_test").unwrap();

        // Assert: inspect internal state through the events log
        let alloc_event = ctx.events.iter().find(|e| {
            e.kind == ResourceKind::Gpr && e.instance == idx && e.event_type == ResourceEventType::Allocate
        }).unwrap();
        assert_eq!(alloc_event.purpose, "purpose_test");
        assert_eq!(alloc_event.instr_idx, 77);

        // Cleanup
        ctx.release(ResourceKind::Gpr, idx);
    }

    #[test]
    fn declare_smem_accumulates_and_reflects_in_available() {
        // Arrange: on GPU profiles, verify cumulative SMEM reduces available
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let smem_cap = ctx.capacity(ResourceKind::SharedMem);
        if smem_cap == 0 {
            return; // skip on CPU
        }

        // Act
        ctx.declare_smem_usage(1024);
        ctx.declare_smem_usage(512);

        // Assert
        assert_eq!(ctx.mem_used(ResourceKind::SharedMem), 1536);
        assert_eq!(ctx.mem_available(ResourceKind::SharedMem), smem_cap - 1536);
    }

    #[test]
    fn usage_report_utilization_zero_when_no_capacity() {
        // Arrange: for any kind with 0 capacity, utilization should be 0.0
        let profile = make_profile();
        let ctx = JitContext::new(&profile);

        // Act
        let report = ctx.usage_report();

        // Assert
        for &kind in &ResourceKind::ALL {
            if ctx.capacity(kind) == 0 {
                assert_eq!(
                    report.utilization[&kind], 0.0,
                    "{:?} with 0 capacity should have 0.0 utilization",
                    kind
                );
            }
        }
    }

    #[test]
    fn suggestion_for_all_kinds_non_empty() {
        // Arrange & Act & Assert: every ResourceKind should have a non-empty suggestion
        let kinds = ResourceKind::ALL;
        for &kind in &kinds {
            let suggestion = JitContext::suggestion_for(kind);
            assert!(!suggestion.is_empty(), "{:?} should have a non-empty suggestion", kind);
        }
    }

    #[test]
    fn release_untracked_kind_panics() {
        // Arrange: on CPU, SharedMem has 0 capacity and is not in the resources HashMap
        let profile = make_profile();
        let ctx = JitContext::new(&profile);

        if ctx.capacity(ResourceKind::SharedMem) == 0 {
            // Act & Assert: releasing an instance of an untracked kind should panic
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut ctx = ctx;
                ctx.release(ResourceKind::SharedMem, 0);
            }));
            assert!(result.is_err(), "releasing untracked resource kind should panic");
        }
    }

    // ── New tests (wave-12klf, +10) ───────────────────────────────────

    #[test]
    fn budget_accessor_returns_correct_reference() {
        // Arrange
        let profile = make_profile();
        let ctx = JitContext::new(&profile);

        // Act
        let budget = ctx.budget();

        // Assert: budget should reflect the same capacities as ctx.capacity()
        for &kind in &ResourceKind::ALL {
            assert_eq!(budget.capacity(kind), ctx.capacity(kind));
        }
    }

    #[test]
    fn advance_to_same_value_is_idempotent() {
        // Arrange
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.advance_to(50);

        // Act: advance to the same value (monotonic, equal is allowed)
        ctx.advance_to(50);

        // Assert: should not panic, current_instr stays at 50
        assert_eq!(ctx.current_instr, 50);

        // Advance forward still works
        ctx.advance_to(100);
        assert_eq!(ctx.current_instr, 100);
    }

    #[test]
    fn freed_slot_is_reused_on_next_allocate() {
        // Arrange: allocate two GPRs, free the first, then allocate again
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        let idx0 = ctx.allocate(ResourceKind::Gpr, "first").unwrap();
        let _idx1 = ctx.allocate(ResourceKind::Gpr, "second").unwrap();
        ctx.release(ResourceKind::Gpr, idx0);

        // Act
        let reused = ctx.allocate(ResourceKind::Gpr, "reused").unwrap();

        // Assert: the freed slot should be reused (linear scan picks first Free)
        assert_eq!(reused, idx0);
    }

    #[test]
    fn resource_state_equality_semantics() {
        // Arrange & Assert: Free == Free
        assert_eq!(ResourceState::Free, ResourceState::Free);

        // Live states with same fields are equal
        assert_eq!(
            ResourceState::Live { purpose: "a", alloc_instr: 1 },
            ResourceState::Live { purpose: "a", alloc_instr: 1 },
        );

        // Live != Free
        assert_ne!(
            ResourceState::Live { purpose: "a", alloc_instr: 1 },
            ResourceState::Free,
        );

        // Different purpose or alloc_instr => not equal
        assert_ne!(
            ResourceState::Live { purpose: "a", alloc_instr: 1 },
            ResourceState::Live { purpose: "b", alloc_instr: 1 },
        );
        assert_ne!(
            ResourceState::Live { purpose: "a", alloc_instr: 1 },
            ResourceState::Live { purpose: "a", alloc_instr: 2 },
        );
    }

    #[test]
    fn independent_tracking_across_resource_kinds() {
        // Arrange: allocate from Gpr and SimdVec independently
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);

        // Act
        let g = ctx.allocate(ResourceKind::Gpr, "gpr_a").unwrap();
        let _v = ctx.allocate(ResourceKind::SimdVec, "vec_a").unwrap();
        let g2 = ctx.allocate(ResourceKind::Gpr, "gpr_b").unwrap();

        // Assert: each kind tracks live count independently
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 2);
        assert_eq!(ctx.live_count(ResourceKind::SimdVec), 1);
        assert_eq!(ctx.peak(ResourceKind::Gpr), 2);
        assert_eq!(ctx.peak(ResourceKind::SimdVec), 1);

        // Release one GPR does not affect SimdVec
        ctx.release(ResourceKind::Gpr, g);
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 1);
        assert_eq!(ctx.live_count(ResourceKind::SimdVec), 1);

        ctx.release(ResourceKind::Gpr, g2);
        assert_eq!(ctx.live_count(ResourceKind::Gpr), 0);
    }

    #[test]
    fn resource_budget_capacity_maps_all_nine_kinds() {
        // Arrange
        let profile = make_profile();
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // Act & Assert: every ResourceKind variant resolves without panic
        // and returns a non-negative value (some may be 0 on CPU)
        for &kind in &ResourceKind::ALL {
            let cap = budget.capacity(kind);
            // Just verify it doesn't panic and is consistent
            assert_eq!(cap, budget.capacity(kind));
        }

        // GPR and SimdVec must be > 0 on any CPU profile
        assert!(budget.capacity(ResourceKind::Gpr) > 0);
        assert!(budget.capacity(ResourceKind::SimdVec) > 0);
    }

    #[test]
    fn update_stack_budget_can_shrink() {
        // Arrange: set a large budget then shrink it
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        ctx.update_stack_budget(16384);
        assert_eq!(ctx.capacity(ResourceKind::Stack), 16384);

        // Act: shrink the stack budget
        ctx.update_stack_budget(2048);

        // Assert
        assert_eq!(ctx.capacity(ResourceKind::Stack), 2048);
        assert_eq!(ctx.budget().stack_bytes, 2048);
    }

    #[test]
    fn snapshot_peak_reflects_historical_max_not_current() {
        // Arrange: allocate 3, release all, allocate 1, then snapshot
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let cap = ctx.capacity(ResourceKind::Gpr);
        if cap < 3 {
            return;
        }

        let mut allocated = Vec::new();
        for _ in 0..3 {
            allocated.push(ctx.allocate(ResourceKind::Gpr, "peak_hist").unwrap());
        }
        for &idx in &allocated {
            ctx.release(ResourceKind::Gpr, idx);
        }
        let _one = ctx.allocate(ResourceKind::Gpr, "final").unwrap();

        // Act
        let snap = ctx.snapshot();

        // Assert: peak in snapshot should be 3 (historical max), not 1 (current live)
        assert_eq!(snap.peak[&ResourceKind::Gpr], 3);
        assert_eq!(snap.live[&ResourceKind::Gpr], 1);
    }

    #[test]
    fn resource_exhausted_is_std_error() {
        // Arrange
        let err = ResourceExhausted {
            kind: ResourceKind::Gpr,
            requested: 1,
            capacity: 16,
            current_live: 16,
            peak: 16,
            suggestion: "reduce register pressure",
        };

        // Act: cast to std::error::Error trait object
        let _: &dyn std::error::Error = &err;

        // Assert: Display output matches specific format
        let msg = format!("{}", err);
        assert!(msg.starts_with("Gpr"));
    }

    #[test]
    fn declare_tmem_usage_reflects_in_available() {
        // Arrange: on GPU profiles, verify TMEM declaration reduces available
        let profile = make_profile();
        let mut ctx = JitContext::new(&profile);
        let tmem_cap = ctx.capacity(ResourceKind::TensorMem);
        if tmem_cap == 0 {
            return; // skip on non-GPU
        }

        // Act
        ctx.declare_tmem_usage(4096);

        // Assert
        assert_eq!(ctx.mem_used(ResourceKind::TensorMem), 4096);
        assert_eq!(ctx.mem_available(ResourceKind::TensorMem), tmem_cap - 4096);

        // Additional declaration accumulates
        ctx.declare_tmem_usage(1024);
        assert_eq!(ctx.mem_used(ResourceKind::TensorMem), 5120);
        assert_eq!(ctx.mem_available(ResourceKind::TensorMem), tmem_cap - 5120);
    }

    // ── JCTX-023: HardwareProfile × ResourceKind cross-matrix tests ──

    fn make_cuda_profile(sm: u32) -> IsaProfile {
        IsaProfile::cuda(sm)
    }

    fn make_hip_profile(gfx: u32) -> IsaProfile {
        IsaProfile::hip(gfx)
    }

    fn make_aarch64_profile() -> IsaProfile {
        IsaProfile::aarch64(true, true, 128, true, true, true)
    }

    fn make_x86_avx2_profile() -> IsaProfile {
        IsaProfile::from_device_profile(&DeviceProfile::detect())
    }

    /// Construct IsaProfile from HardwareProfile via its platform() mapping.
    /// For CPU profiles, we use the real DeviceProfile and rely on HardwareProfile::ALL
    /// for GPU/ARM profiles that require constructed IsaProfiles.
    fn isa_profile_for_hw(hw: &super::super::hardware_profile::HardwareProfile) -> IsaProfile {
        use super::super::hardware_profile::HardwareProfile;
        match hw {
            HardwareProfile::CudaSM80 => make_cuda_profile(80),
            HardwareProfile::CudaSM90 => make_cuda_profile(90),
            HardwareProfile::CudaSM100 => make_cuda_profile(100),
            HardwareProfile::RocmMI200 => make_hip_profile(908),
            HardwareProfile::RocmMI300 => make_hip_profile(942),
            HardwareProfile::CpuAvx2 | HardwareProfile::CpuAvx512 | HardwareProfile::CpuAvx10_2 => {
                let dp = DeviceProfile::detect();
                IsaProfile::from_device_profile(&dp)
            }
            HardwareProfile::AppleM1 | HardwareProfile::AppleM2 | HardwareProfile::AppleM3 => {
                // Metal profile: construct via platform() for budget derivation
                let platform = hw.platform();
                IsaProfile {
                    platform,
                    gpr_regs: vec![],
                    scratch_gprs: vec![],
                    vec_regs: vec![],
                    scratch_vec_regs: vec![],
                    tile_regs: vec![],
                    mask_regs: vec![],
                    abi: super::super::codegen::vm::isa_profile::AbiConvention {
                        arg_regs: vec![], stack_arg_offset: 0, callee_saved: vec![],
                        caller_saved: vec![], callee_saved_vec: vec![],
                        stack_alignment: 0, red_zone_bytes: 0,
                    },
                    cache: super::super::codegen::vm::isa_profile::CacheHierarchy {
                        l1d_bytes: 0, l1i_bytes: 0, l2_bytes: 0, l3_bytes: 0,
                        cacheline_bytes: 128, tmem_bytes: 0, smem_bytes: 32 * 1024, lds_bytes: 0,
                    },
                    features: vec![],
                    k_unroll_factor: 4,
                    dot_cap: crate::dispatch::device_profile::DotProductCap::SimdAssisted,
                }
            }
            HardwareProfile::ArmNeoverse => make_aarch64_profile(),
            HardwareProfile::Generic => {
                let dp = DeviceProfile::detect();
                IsaProfile::from_device_profile(&dp)
            }
        }
    }

    #[test]
    fn test_resource_budget_matrix_all_profiles() {
        use super::super::hardware_profile::HardwareProfile;

        let mut tested = 0;
        for hw in &HardwareProfile::ALL {
            let profile = isa_profile_for_hw(hw);
            let platform = &profile.platform;
            let ctx = JitContext::new(&profile);
            let budget = ctx.budget();

            for &kind in &ResourceKind::ALL {
                if !kind.is_available_on(platform) {
                    // Unavailable resources should have 0 capacity
                    assert_eq!(
                        budget.capacity(kind), 0,
                        "{hw:?} × {kind:?}: unavailable resource should have 0 capacity"
                    );
                    continue;
                }
                // Available resources should have positive capacity
                let cap = budget.capacity(kind);
                assert!(
                    cap > 0,
                    "{hw:?} × {kind:?}: available resource must have budget > 0, got {}",
                    cap
                );
                tested += 1;
            }
        }
        // Sanity: we should have tested a meaningful number of combinations
        assert!(tested >= 25, "expected >= 25 valid combinations, got {}", tested);
    }

    #[test]
    fn test_resource_kind_is_available_on_smoke() {
        use super::super::codegen::vm::isa_profile::Platform;

        // Gpr/SimdVec/Stack available on CPU only
        let x86 = Platform::X86_64 { has_avx512: false, has_bf16: false, has_vnni: false,
            has_avx512fp16: false, has_amx: false, has_amx_fp16: false,
            has_amx_complex: false, has_amx_transpose: false, has_amx_fp8: false,
            has_avx10_2: false, has_apx: false, has_vp2intersect: false };
        assert!(ResourceKind::Gpr.is_available_on(&x86));
        assert!(ResourceKind::SimdVec.is_available_on(&x86));
        assert!(ResourceKind::Stack.is_available_on(&x86));

        let cuda90 = Platform::Cuda { sm_version: 90, warp_size: 32, shared_mem_kb: 228,
            reg_file_per_sm: 65536, max_regs_per_thread: 255,
            has_wgmma: true, has_tma: true, has_warp_spec: true, has_fp8: true,
            has_tmem: false, has_block_scaled: false, has_native_fp4: false,
            has_native_fp6: false, has_cluster: false, has_2cta_mma: false, tmem_size_kb: 0 };
        assert!(!ResourceKind::Gpr.is_available_on(&cuda90));
        assert!(!ResourceKind::SimdVec.is_available_on(&cuda90));
        assert!(!ResourceKind::Stack.is_available_on(&cuda90));

        // SharedMem: GPU-only
        let x86 = Platform::X86_64 { has_avx512: false, has_bf16: false, has_vnni: false,
            has_avx512fp16: false, has_amx: false, has_amx_fp16: false,
            has_amx_complex: false, has_amx_transpose: false, has_amx_fp8: false,
            has_avx10_2: false, has_apx: false, has_vp2intersect: false };
        assert!(!ResourceKind::SharedMem.is_available_on(&x86));

        // TensorMem: SM100+ only
        let cuda90 = Platform::Cuda { sm_version: 90, warp_size: 32, shared_mem_kb: 228,
            reg_file_per_sm: 65536, max_regs_per_thread: 255,
            has_wgmma: true, has_tma: true, has_warp_spec: true, has_fp8: true,
            has_tmem: false, has_block_scaled: false, has_native_fp4: false,
            has_native_fp6: false, has_cluster: false, has_2cta_mma: false, tmem_size_kb: 0 };
        assert!(!ResourceKind::TensorMem.is_available_on(&cuda90));

        let cuda100 = Platform::Cuda { sm_version: 100, warp_size: 32, shared_mem_kb: 228,
            reg_file_per_sm: 65536, max_regs_per_thread: 255,
            has_wgmma: true, has_tma: true, has_warp_spec: true, has_fp8: true,
            has_tmem: true, has_block_scaled: true, has_native_fp4: true,
            has_native_fp6: true, has_cluster: true, has_2cta_mma: true, tmem_size_kb: 256 };
        assert!(ResourceKind::TensorMem.is_available_on(&cuda100));
    }

    #[test]
    fn test_matrix_cuda_sm80_budget() {
        let profile = make_cuda_profile(80);
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // SM80: SMEM >= 164KB, no TMEM, no barriers (no warp_spec)
        assert!(budget.shared_mem_bytes >= 164 * 1024, "SM80 SMEM >= 164KB");
        assert_eq!(budget.tensor_mem_bytes, 0, "SM80 no TMEM");
        // SM80 has no warp_spec so no barriers
        assert_eq!(budget.barrier_total, 0, "SM80 no barriers");
    }

    #[test]
    fn test_matrix_cuda_sm90_budget() {
        let profile = make_cuda_profile(90);
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // SM90: SMEM >= 228KB, no TMEM, has barriers (warp_spec), 4 tile accumulators
        assert!(budget.shared_mem_bytes >= 228 * 1024, "SM90 SMEM >= 228KB");
        assert_eq!(budget.tensor_mem_bytes, 0, "SM90 no TMEM");
        assert!(budget.barrier_total >= 2, "SM90 has warp_spec barriers");
        assert!(budget.tile_accumulator_total >= 4, "SM90 has WGMMA accumulators");
    }

    #[test]
    fn test_matrix_cuda_sm100_budget() {
        let profile = make_cuda_profile(100);
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // SM100: SMEM >= 228KB, TMEM >= 256KB/SM, barriers, 4 tile accumulators
        assert!(budget.shared_mem_bytes >= 228 * 1024, "SM100 SMEM >= 228KB");
        assert!(budget.tensor_mem_bytes >= 256 * 1024, "SM100 TMEM >= 256KB");
        assert!(budget.barrier_total >= 2, "SM100 has barriers");
        assert!(budget.tile_accumulator_total >= 4, "SM100 has tcgen05 accumulators");
    }

    #[test]
    fn test_matrix_rocm_mi200_budget() {
        let profile = make_hip_profile(908);
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // MI200: LDS = 64KB, no TMEM, 4 MFMA accumulators, no barriers
        assert_eq!(budget.shared_mem_bytes, 64 * 1024, "MI200 LDS = 64KB");
        assert_eq!(budget.tensor_mem_bytes, 0, "MI200 no TMEM");
        assert!(budget.tile_accumulator_total >= 4, "MI200 has MFMA accumulators");
    }

    #[test]
    fn test_matrix_rocm_mi300_budget() {
        let profile = make_hip_profile(942);
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // MI300: LDS = 64KB, no TMEM, 4 MFMA accumulators
        assert_eq!(budget.shared_mem_bytes, 64 * 1024, "MI300 LDS = 64KB");
        assert_eq!(budget.tensor_mem_bytes, 0, "MI300 no TMEM");
        assert!(budget.tile_accumulator_total >= 4, "MI300 has MFMA accumulators");
    }

    #[test]
    fn test_matrix_aarch64_sme2_budget() {
        let profile = make_aarch64_profile();
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // AArch64 with SME2: 29 allocatable GPRs (31 - scratch), 21 allocatable vec regs,
        // 7 predicate masks, 4 tile regs
        assert!(budget.gpr_total > 0, "AArch64 has GPRs");
        assert!(budget.simd_vec_total > 0, "AArch64 has vec regs");
        assert!(budget.predicate_total >= 7, "AArch64 SVE has >= 7 mask regs");
        assert!(budget.tile_total >= 4, "AArch64 SME has >= 4 tile regs");
        // No GPU resources
        assert_eq!(budget.shared_mem_bytes, 0, "AArch64 no SMEM");
        assert_eq!(budget.tensor_mem_bytes, 0, "AArch64 no TMEM");
    }

    #[test]
    fn test_matrix_cpu_gpr_budget() {
        let profile = make_x86_avx2_profile();
        let budget = JitResourceBudget::from_isa_profile(&profile);

        // x86: GPR >= 11 (16 - 2 frame - 3 scratch), SimdVec >= 10 (16 - 6 scratch)
        assert!(budget.gpr_total >= 11, "x86 GPR >= 11, got {}", budget.gpr_total);
        assert!(budget.simd_vec_total >= 10, "x86 SimdVec >= 10, got {}", budget.simd_vec_total);
        assert!(budget.stack_bytes > 0, "x86 stack must be > 0");
        // No GPU resources
        assert_eq!(budget.shared_mem_bytes, 0, "x86 no SMEM");
        assert_eq!(budget.tensor_mem_bytes, 0, "x86 no TMEM");
        assert_eq!(budget.barrier_total, 0, "x86 no barriers");
    }

    #[test]
    fn test_matrix_predicate_avx512_vs_avx2() {
        // AVX-512 has predicate (mask) registers; AVX2 does not
        let profile = make_x86_avx2_profile();
        let budget = JitResourceBudget::from_isa_profile(&profile);

        if matches!(&profile.platform, Platform::X86_64 { has_avx512: true, .. }) {
            assert!(budget.predicate_total >= 8, "AVX-512 has 8 mask regs");
        } else {
            assert_eq!(budget.predicate_total, 0, "AVX2 has no mask regs");
        }
    }

    #[test]
    fn test_matrix_tile_amx_vs_non_amx() {
        let profile = make_x86_avx2_profile();
        let budget = JitResourceBudget::from_isa_profile(&profile);

        if matches!(&profile.platform, Platform::X86_64 { has_amx: true, .. }) {
            assert!(budget.tile_total >= 8, "AMX has 8 tile regs");
        } else {
            assert_eq!(budget.tile_total, 0, "No AMX = no tile regs");
        }
    }

    #[test]
    fn test_matrix_shared_mem_sm90_allocate_release() {
        let profile = make_cuda_profile(90);
        let mut ctx = JitContext::new(&profile);
        let smem_cap = ctx.capacity(ResourceKind::SharedMem);
        if smem_cap == 0 { return; }

        // Declare SMEM usage and verify tracking
        ctx.declare_smem_usage(64 * 1024);
        assert_eq!(ctx.mem_used(ResourceKind::SharedMem), 64 * 1024);
        assert!(ctx.mem_available(ResourceKind::SharedMem) >= smem_cap - 64 * 1024);
    }

    #[test]
    fn test_matrix_tmem_sm100_allocate_declare() {
        let profile = make_cuda_profile(100);
        let mut ctx = JitContext::new(&profile);
        let tmem_cap = ctx.capacity(ResourceKind::TensorMem);
        if tmem_cap == 0 { return; }

        // TMEM should be >= 256KB on SM100
        assert!(tmem_cap >= 256 * 1024, "SM100 TMEM >= 256KB, got {}", tmem_cap);

        ctx.declare_tmem_usage(128 * 1024);
        assert_eq!(ctx.mem_used(ResourceKind::TensorMem), 128 * 1024);
        assert_eq!(ctx.mem_available(ResourceKind::TensorMem), tmem_cap - 128 * 1024);
    }

    #[test]
    fn test_matrix_barrier_cuda_sm90() {
        let profile = make_cuda_profile(90);
        let mut ctx = JitContext::new(&profile);
        let barrier_cap = ctx.capacity(ResourceKind::Barrier);
        if barrier_cap == 0 { return; }

        // SM90 has warp_spec barriers — should be allocatable
        let idx = ctx.allocate(ResourceKind::Barrier, "sync_point").unwrap();
        assert_eq!(ctx.live_count(ResourceKind::Barrier), 1);
        ctx.release(ResourceKind::Barrier, idx);
        assert_eq!(ctx.live_count(ResourceKind::Barrier), 0);
    }

    #[test]
    fn test_matrix_unavailable_kinds_have_zero_budget() {
        use super::super::hardware_profile::HardwareProfile;

        for hw in &HardwareProfile::ALL {
            let profile = isa_profile_for_hw(hw);
            let budget = JitResourceBudget::from_isa_profile(&profile);

            for &kind in &ResourceKind::ALL {
                if !kind.is_available_on(&profile.platform) {
                    assert_eq!(
                        budget.capacity(kind), 0,
                        "{hw:?} × {kind:?}: unavailable kind should have 0 budget"
                    );
                }
            }
        }
    }

    #[test]
    fn test_matrix_no_double_allocate_beyond_budget() {
        // Verify that each available resource kind correctly rejects allocation beyond capacity
        let profile = make_cuda_profile(90);
        let mut ctx = JitContext::new(&profile);

        for &kind in &ResourceKind::ALL {
            if !kind.is_available_on(&profile.platform) { continue; }
            let cap = ctx.capacity(kind);
            if cap == 0 || kind.is_memory() { continue; } // Skip memory-type resources

            // Exhaust all instances
            let mut allocated = Vec::new();
            for _ in 0..cap {
                allocated.push(ctx.allocate(kind, "exhaust").unwrap());
            }

            // One more should fail
            let result = ctx.allocate(kind, "overflow");
            assert!(result.is_err(), "{kind:?}: allocation beyond capacity should fail");

            // Release all
            for idx in allocated {
                ctx.release(kind, idx);
            }
        }
    }

    #[test]
    fn test_hardware_profile_all_count() {
        use super::super::hardware_profile::HardwareProfile;
        assert_eq!(HardwareProfile::ALL.len(), 12);
    }

    #[test]
    fn test_hardware_profile_platform_mapping_consistency() {
        use super::super::hardware_profile::HardwareProfile;

        for hw in &HardwareProfile::ALL {
            let platform = hw.platform();
            // GPU profiles should map to GPU platforms
            match hw {
                HardwareProfile::CudaSM80 | HardwareProfile::CudaSM90 | HardwareProfile::CudaSM100 => {
                    assert!(matches!(platform, Platform::Cuda { .. }), "{hw:?} should map to Cuda");
                }
                HardwareProfile::RocmMI200 | HardwareProfile::RocmMI300 => {
                    assert!(matches!(platform, Platform::Hip { .. }), "{hw:?} should map to Hip");
                }
                HardwareProfile::CpuAvx2 | HardwareProfile::CpuAvx512 | HardwareProfile::CpuAvx10_2 => {
                    assert!(matches!(platform, Platform::X86_64 { .. }), "{hw:?} should map to X86_64");
                }
                HardwareProfile::AppleM1 | HardwareProfile::AppleM2 | HardwareProfile::AppleM3 => {
                    assert!(matches!(platform, Platform::Metal { .. }), "{hw:?} should map to Metal");
                }
                HardwareProfile::ArmNeoverse => {
                    assert!(matches!(platform, Platform::AArch64 { .. }), "{hw:?} should map to AArch64");
                }
                HardwareProfile::Generic => {} // fallback, no strong assertion
            }
        }
    }
}
