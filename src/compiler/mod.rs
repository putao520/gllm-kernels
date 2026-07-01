//! Layer 3: Inference Compiler — JIT compilation of transformer layers.
//!
//! The compiler takes a `ModelConfig`, builds a `LayerIR` for each layer,
//! plans execution via `ExecutionPlan`, generates machine code, and caches
//! the result.
//!
//! # Pipeline
//!
//! ```text
//! LayerIR → CompilerGraph → Scalar + SymExec (ScalarOpRegistry + OpTrace via SymbolicExecutor)
//!         → SemanticDAG (OpClass auto-derivation from OpTrace::ComputePattern)
//!         → Fusion (fuse_with_dag + HW constraints + Parallel strategy + Buffer alloc)
//!         → ISA Lowering (CodeGen: x86_64/aarch64 JIT via iced-x86/dynasm-rs)
//!         → CompiledLayer (cached)
//! ```
//!
//! Scalar + SymExec extracts `OpTrace` from `extern "C"` scalar functions via binary
//! symbolic execution. SemanticDAG builds a `SemanticDAG` with auto-derived
//! `OpClass`. Fusion performs fusion decisions with HW constraint checks,
//! parallel strategy selection, and interval-graph buffer allocation.
//! ISA Lowering generates native machine code from `TraceOp` sequences.

use std::collections::HashMap;

pub mod ir;
pub mod planner;
pub mod executable;
pub mod cache;
pub mod codegen;
pub mod graph;
pub mod rope_scaling;
pub mod semantics;
pub mod fusion;
pub mod fwht_fusion;
pub mod trace;
pub mod registry;
pub mod backend_cap;
pub mod semantic_dag;
pub mod symexec;
pub mod hw_constraints;
pub mod buffer_alloc;
pub mod model_adapter;
pub mod hotpatch;
pub mod hardware_profile;
pub mod jit_context;
pub mod pain_point;
pub mod accel_registry;
pub mod layout_negotiator;
pub mod virtual_tensor;
pub mod group_dep;
pub mod virtual_activation;
pub mod pack_map;
pub mod counters;
pub mod resource_estimator;
pub mod parallel_compile;

pub mod dump;
pub mod diagnostics;
pub mod dtype_chain;
pub mod graph_geometry;
pub mod mega_kernel_abi;
pub mod quant_ir;
pub mod quant_convert;
pub mod quant_format;
pub mod quant_pipeline;
#[cfg(test)]
mod quant_activation_test;

// ── Public API re-exports (used by external consumers: gllm) ───────────
pub use executable::CompiledLayer;
pub use mega_kernel_abi::{
    MegaKernelFn, KernelWeightLayout, BufferLayout,
    PerLayerWeightLayout,
    BusinessConfig, MtpKernelConfig, OutputMode, PoolMode, SgConfig, CotStepConfig,
    CompileConfig, CompileTarget,
    HeteroLayerConfig, HeteroKernelWeightLayout,
    MEGA_KERNEL_PARAMS, MEGA_KERNEL_STACK_OFFSETS,
};

/// Output of `compile_mega_kernel()`: compiled layer code + layout metadata.
pub struct MegaKernelCompileOutput {
    /// JIT-compiled code for a single layer template.
    /// Will be wrapped in a layer loop at codegen time (ISA Lowering).
    pub layer_code: CompiledLayer,
    /// Weight blob layout for the full model.
    pub weight_layout: KernelWeightLayout,
    /// Runtime buffer layout (activations, logits, sampling workspace).
    pub buffer_layout: BufferLayout,
    /// Number of layers (layer loop bound).
    pub num_layers: usize,
    /// Vocabulary size (embedding + logits-producer dimensions).
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden: usize,
    /// RoPE cos/sin cache requirement (caller must fill scratchpad before each call).
    pub rope_cache: Option<codegen::RopeCacheRequirement>,
    /// Total scratchpad bytes needed (intermediate tensors + RoPE cache + logits).
    pub total_scratchpad_bytes: usize,
    /// Logits region offset within scratchpad (after intermediates + RoPE cache).
    /// Rust argmax reads from scratchpad[logits_scratch_offset + (seq-1)*vocab_bytes..].
    pub logits_scratch_offset: usize,
    /// Heterogeneous layer weight layout (for models like Gemma-4 E2B with
    /// alternating sliding/full attention layers). None for homogeneous models.
    pub hetero_layout: Option<mega_kernel_abi::HeteroKernelWeightLayout>,
    /// JIT source map — 仅当 debug_jit=true 时生成。
    /// 包含 VmInstr → 机器码偏移 → Op 标签的映射，供 DAP 调试器使用。
    pub source_map: Option<codegen::vm::debug_map::JitSourceMap>,
    /// BCE-20260629-006: Intermediate tensor sources (TensorId → TensorPtrSource)
    /// 供 DIAG harness 动态查询 intermediate tensor 的 scratchpad offset。
    /// 之前 compile 内部丢弃了 BufferAllocation.tensor_sources，现在透出。
    pub tensor_sources: HashMap<crate::compiler::graph::TensorId, crate::compiler::buffer_alloc::TensorPtrSource>,
}

/// Output of GPU mega-kernel compilation (PTX/HIP/MSL source).
#[derive(Debug)]
pub struct GpuMegaKernelOutput {
    /// GPU kernel source code (PTX/HIP/MSL text).
    pub gpu_code: Vec<u8>,
    /// Required scratchpad size in bytes.
    pub total_scratchpad_bytes: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden: usize,
    /// RoPE cache requirements.
    pub rope_cache: Option<codegen::RopeCacheRequirement>,
    /// Offset of logits region in scratchpad.
    pub logits_scratch_offset: usize,
}

/// Output of forward-only GPU kernel compilation (PTX/HIP/MSL source).
#[derive(Debug)]
pub struct GpuForwardOutput {
    /// GPU kernel source code (PTX/HIP/MSL text).
    pub gpu_code: Vec<u8>,
    /// Required scratchpad size in bytes.
    pub total_scratchpad_bytes: usize,
}

/// SPEC REQ-UMK-001: Unified compilation output.
///
/// `InferenceCompiler::compile()` returns this enum, with the variant selected
/// by `CompileConfig::target`. CPU target produces `Cpu` (mega-kernel native
/// code + layouts); GPU target produces `Gpu` (PTX/HIP/MSL source).
pub enum CompileOutput {
    /// CPU mega-kernel compilation result (JIT-compiled native code + layouts).
    Cpu(MegaKernelCompileOutput),
    /// GPU mega-kernel compilation result (PTX/HIP/MSL source).
    Gpu(GpuMegaKernelOutput),
}

impl CompileOutput {
    /// Unwrap the CPU variant. Panics with a descriptive message if this is
    /// a GPU output.
    pub fn expect_cpu(self) -> MegaKernelCompileOutput {
        match self {
            CompileOutput::Cpu(out) => out,
            CompileOutput::Gpu(_) => panic!("expected CompileOutput::Cpu, got Gpu"),
        }
    }

    /// Unwrap the GPU variant. Panics with a descriptive message if this is
    /// a CPU output.
    pub fn expect_gpu(self) -> GpuMegaKernelOutput {
        match self {
            CompileOutput::Gpu(out) => out,
            CompileOutput::Cpu(_) => panic!("expected CompileOutput::Gpu, got Cpu"),
        }
    }

    /// Returns true if this is the CPU variant.
    pub fn is_cpu(&self) -> bool {
        matches!(self, CompileOutput::Cpu(_))
    }

    /// Returns true if this is the GPU variant.
    pub fn is_gpu(&self) -> bool {
        matches!(self, CompileOutput::Gpu(_))
    }
}

pub use graph::{CompilerGraph, Op, RopeScaling, TensorId, WeightLayout, SymDim, ShapeBinding};
pub use ir::MoeConfig;
pub use rope_scaling::{compute_attention_scaling, compute_inv_freq, fill_cos_sin_table, fill_cos_sin_table_partial};
pub use registry::ScalarOpRegistry;
pub use semantic_dag::{SemanticDAG, Boundness};
pub use fusion::FusionPlan;
pub use planner::ExecutionPlan;
pub use planner::HwOptPlan;
pub use planner::{
    RooflineAnalyzer, CacheBudgetSolver, GemmSolver, AttentionSolver,
    FusionSolver, ParallelismSolver, BatchSolver, FeatureRouter,
};
pub use crate::types::CompilerError;

// ── Internal re-exports (pub within crate, not part of public API) ────
pub(crate) use ir::LayerIR;
pub(crate) use cache::CompilationCache;

use crate::dispatch::{DeviceProfile, device_profile};
use crate::types::InferenceError;

/// Global lock serializing JIT compilation and execution.
///
/// The codegen pipeline (fusion → buffer alloc → VM lowering → ISA emission)
/// accesses thread-internal state that can be corrupted when multiple threads
/// compile and execute JIT code concurrently (observed as NaN in output buffers).
/// This lock ensures only one compile+execute sequence runs at a time.
///
/// This is a conservative fix; the root cause is likely in x86 JIT code
/// accessing some implicitly-shared resource during execution.
static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard returned by `compile_lock()`. Releases the global JIT lock on drop.
pub struct CompileGuard<'a>(std::sync::MutexGuard<'a, ()>);

/// Acquire the global JIT compile+execute lock.
///
/// SPEC REQ-UMK-001: This is now `pub(crate)` — the unified `compile()` entry
/// point acquires it internally. External callers must not call this directly.
pub(crate) fn compile_lock() -> CompileGuard<'static> {
    CompileGuard(COMPILE_LOCK.lock().unwrap())
}

/// The inference compiler: compiles transformer layers to native code.
///
/// Holds the compilation cache and device profile. Typically created once
/// at model load time and shared across inference calls.
pub struct InferenceCompiler {
    profile: DeviceProfile,
    cache: CompilationCache,
}

impl InferenceCompiler {
    /// Create a new compiler with the detected hardware profile.
    pub fn new() -> Self {
        InferenceCompiler {
            profile: device_profile().clone(),
            cache: CompilationCache::default_disk(),
        }
    }

    /// Create a compiler with a specific profile (for testing).
    pub fn with_profile(profile: DeviceProfile) -> Self {
        InferenceCompiler {
            profile,
            cache: CompilationCache::default_disk(),
        }
    }

    /// Access the device profile driving compilation decisions.
    ///
    /// REQ-API-10: The profile is the single source of hardware dispatch
    /// information. All ISA-specific code generation is driven by this
    /// profile — runtime branches on hardware capabilities are prohibited.
    pub fn device_profile(&self) -> &DeviceProfile {
        &self.profile
    }

    /// Clear the in-memory compilation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Clear only the on-disk cache, keeping memory entries.
    pub fn clear_disk_cache(&mut self) {
        self.cache.clear_disk_cache();
    }

    /// Number of cached compilations.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Print compilation resource report (code size, cache stats).
    ///
    /// Debug-only: outputs to stderr when GLLM_DEBUG_RESOURCE=1.
    pub fn print_resource_report(&self) {
        if std::env::var("GLLM_DEBUG_RESOURCE").as_deref() == Ok("1") {
            eprintln!("[ResourceReport] cache_entries={} isa={:?}",
                self.cache.len(),
                self.profile.isa,
            );
        }
    }

    fn compute_hash(&self, ir: &LayerIR) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // MoE configuration
        if let Some(moe) = &ir.moe {
            moe.num_experts.hash(&mut hasher);
            moe.top_k.hash(&mut hasher);
        }

        // Shape parameters
        ir.hidden.hash(&mut hasher);
        ir.num_heads.hash(&mut hasher);
        ir.num_kv_heads.hash(&mut hasher);
        ir.head_dim.hash(&mut hasher);
        ir.intermediate.hash(&mut hasher);

        // Quantization: hash Option discriminant + inner QuantType discriminant
        match &ir.quant {
            None => 0u8.hash(&mut hasher),
            Some(qt) => {
                1u8.hash(&mut hasher);
                std::mem::discriminant(qt).hash(&mut hasher);
            }
        }

        // DType already derives Hash
        ir.dtype.hash(&mut hasher);

        // RoPE / normalization
        ir.rope_theta.to_bits().hash(&mut hasher);
        ir.rms_eps.to_bits().hash(&mut hasher);

        // Batch / sequence limits
        ir.max_batch.hash(&mut hasher);
        ir.max_seq.hash(&mut hasher);

        // Partial rotary factor
        ir.partial_rotary_factor.to_bits().hash(&mut hasher);

        // Activation (no Hash derive — use discriminant)
        std::mem::discriminant(&ir.activation).hash(&mut hasher);

        // Hardware fingerprint
        self.profile.hw_info.fingerprint().hash(&mut hasher);

        hasher.finish()
    }

    /// Full JIT compilation pipeline:
    /// LayerIR → CompilerGraph → Scalar + SymExec (registry) → SemanticDAG
    ///         → Fusion + HW + parallel + buffer → ISA Lowering (codegen)
    ///         → CodegenOutput
    ///
    /// ISA Lowering has an MVP implementation under the `jit-x86` feature flag
    /// (see `codegen::emitter::X86CodeGen`). Without the feature flag,
    /// an error is returned.
    fn jit_compile(&self, ir: &LayerIR) -> Result<codegen::CodegenOutput, InferenceError> {
        // Build ExecutionPlan (HwOptPlan) — the single source of strategy decisions
        let exec_plan = planner::ExecutionPlan::build(ir, &self.profile, &planner::StrategyBias::default());

        // Build CompilerGraph DAG
        let graph = CompilerGraph::from_layer_ir(ir, &self.profile)
            .map_err(InferenceError::CompileError)?;

        // ScalarOpRegistry (OpTrace cache) + SemanticDAG (OpClass auto-derivation)
        let registry = ScalarOpRegistry::with_defaults();
        let semantic_dag = SemanticDAG::from_graph(&graph, &registry);

        // Fusion decisions + HW constraint validation + buffer allocation
        let mut fusion_plan = fusion::fuse_with_dag_prebuilt(&graph, &semantic_dag, &exec_plan, None, None);
        hw_constraints::enforce_constraints(&mut fusion_plan.groups, &graph, &exec_plan);
        let lifetimes = buffer_alloc::analyze_lifetimes(&graph, &fusion_plan, None, None);
        let alloc = buffer_alloc::allocate_buffers_aligned(&lifetimes, self.profile.hw_info.cacheline_bytes, None, None, &graph, None);

        // SPEC/39 REQ-UMK-001: All compilation produces MegaKernelFn ABI code.
        // The 10-param ABI (CompiledLayerFn) is physically deleted.
        let sym_map = codegen::vm::plan_lower::SymDimSlotMap::mega_kernel_abi();

        // ISA Lowering: Code generation
        #[cfg(feature = "jit-x86")]
        {
            codegen::vm::plan_lower::compile_layer_with_sym_map(
                &fusion_plan, &graph, &alloc, &exec_plan, Some(&registry), &sym_map,
            ).map_err(InferenceError::CompileError)
        }

        #[cfg(not(feature = "jit-x86"))]
        {
            let _ = (fusion_plan, alloc, graph, sym_map);
            Err(InferenceError::CompileError("JIT backend not enabled (feature jit-x86 required)".into()))
        }
    }

    /// SPEC REQ-UMK-001: Unified compilation entry point.
    ///
    /// This is the **sole** public compilation method on `InferenceCompiler`.
    /// Replaces the former 6-entry-point API (`compile_mega_kernel_from_graph`,
    /// `compile_mega_kernel_to_gpu`, `compile_model`, `compile_layer`,
    /// `compile_model_incremental`, `compile_lock`).
    ///
    /// Behavior is fully driven by `CompileConfig`:
    /// - `target: CompileTarget::Cpu` (default) → JIT-compile native code,
    ///   returns `CompileOutput::Cpu(MegaKernelCompileOutput)`.
    /// - `target: CompileTarget::Gpu { sm_version }` → emit PTX/HIP/MSL source,
    ///   returns `CompileOutput::Gpu(GpuMegaKernelOutput)`. Requires
    ///   `jit-cuda` or `jit-hip` feature flag.
    ///
    /// All model geometry (hidden, num_heads, vocab_size, etc.) is derived from
    /// the graph's OpKind variants and tensor shapes. Only non-derivable fields
    /// are passed via `CompileConfig`. The internal `compile_lock()` is acquired
    /// automatically; callers do not need to (and cannot) take it.
    pub fn compile(
        &mut self,
        graph: CompilerGraph,
        config: &mega_kernel_abi::CompileConfig,
        hetero_layout: Option<mega_kernel_abi::HeteroKernelWeightLayout>,
    ) -> Result<CompileOutput, InferenceError> {
        // SPEC REQ-UMK-001: compile_lock is now pub(crate) and acquired inside
        // compile() — callers no longer need to (and cannot) take it manually.
        let _guard = compile_lock();

        match config.target {
            mega_kernel_abi::CompileTarget::Cpu => {
                self.compile_cpu(graph, config, hetero_layout)
                    .map(CompileOutput::Cpu)
            }
            mega_kernel_abi::CompileTarget::Gpu { sm_version } => {
                #[cfg(any(feature = "jit-cuda", feature = "jit-hip"))]
                {
                    self.compile_gpu(graph, config, sm_version)
                        .map(CompileOutput::Gpu)
                }
                #[cfg(not(any(feature = "jit-cuda", feature = "jit-hip")))]
                {
                    let _ = (graph, sm_version);
                    Err(InferenceError::CompileError(
                        "GPU compilation requested (CompileTarget::Gpu) but neither jit-cuda nor jit-hip feature is enabled".into(),
                    ))
                }
            }
        }
    }

    /// CPU codegen path — emits native JIT machine code (x86_64/aarch64).
    ///
    /// Internal helper called by `compile()` when `config.target == Cpu`.
    fn compile_cpu(
        &mut self,
        graph: CompilerGraph,
        config: &mega_kernel_abi::CompileConfig,
        hetero_layout: Option<mega_kernel_abi::HeteroKernelWeightLayout>,
    ) -> Result<MegaKernelCompileOutput, InferenceError> {
        let t0 = std::time::Instant::now();
        static COMPILE_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let compile_id = COMPILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        eprintln!("[COMPILE-MEGA] compile(cpu) #{}, ops={}",
            compile_id, graph.ops.len());

        // REQ-FAIL-TRIANG-001: IR-layer pre_check at compile() entry.
        // Validates graph structural invariants before any compilation proceeds.
        let ir_errors = diagnostics::pre_check(&graph);
        if !ir_errors.is_empty() {
            let details: Vec<String> = ir_errors.iter()
                .take(5)
                .map(|e| format!("{}", e))
                .collect();
            return Err(InferenceError::CompileError(
                format!("IR-ERR: {} precondition violation(s).\n{}",
                    ir_errors.len(), details.join("\n")).into()
            ));
        }

        // REQ-DTYPE-CHAIN-005: Dtype chain validation gate.
        // Blocks Mega-Kernel generation if dtype breakpoints are detected.
        let dtype_validation = dtype_chain::DtypeChainValidation::validate(&graph, &DeviceProfile::detect());
        if !dtype_validation.is_valid {
            let details: Vec<String> = dtype_validation.breakpoints.iter()
                .take(5)
                .map(|bp| format!("  op '{}' tensor {:?}: expected {:?}, got {:?} — {}",
                    bp.op_label, bp.tensor_id, bp.expected_dtype, bp.actual_dtype, bp.suggestion))
                .collect();
            return Err(InferenceError::CompileError(format!(
                "DTYPE-CHAIN validation failed: {} breakpoint(s).\n{}",
                dtype_validation.num_breakpoints, details.join("\n")
            ).into()));
        }

        // REQ-BACKEND-CAP-003: Capability matrix gate.
        // Build the capability matrix from ScalarOpRegistry + DeviceProfile ISV
        // and validate all graph ops are supported before proceeding to codegen.
        {
            let registry = ScalarOpRegistry::with_defaults();
            let registered_keys = registry.registered_keys();
            let cap_matrix = backend_cap::BackendCapMatrix::build(&self.profile, &registered_keys);

            // Collect all OpKindKeys from the graph
            let graph_op_keys: Vec<_> = graph.ops.iter()
                .map(|cop| ScalarOpRegistry::key_from_op(&cop.op))
                .collect();

            let profile_label = format!("{:?} {:?}", self.profile.arch, self.profile.isa);
            cap_matrix.validate_graph_ops(&graph_op_keys, self.profile.isa, &profile_label)
                .map_err(|cap_err| {
                    InferenceError::CompileError(CompilerError::CapabilityUnsupported {
                        op_kind: cap_err.op_kind,
                        device_profile: cap_err.device_profile,
                        strategy: cap_err.strategy.to_string(),
                        reason: cap_err.reason,
                    })
                })?;
        }

        // SPEC/39 REQ-UMK-001: single compilation entry point handles all graph topologies.
        // All graphs (with or without layer loops) go through compile_mega_kernel_vm.
        // Simple graphs use LoopBegin { bound: Const(1) } + LoopEnd — single iteration, zero overhead.
        let geometry = graph_geometry::GraphDerivedGeometry::from_graph(&graph, &DeviceProfile::detect())
            .map_err(|e| InferenceError::CompileError(format!("GraphDerivedGeometry: {}", e).into()))?;

        // [FIX-PSC29] Determine SG presence from graph ops (SgDetect/SgInject).
        // When no SG ops exist, BufferLayout skips SG allocation entirely.
        let sg_enabled = graph.ops.iter().any(|op| {
            matches!(op.op_resolved(&graph), Some(graph::Op::SgDetect { .. }) | Some(graph::Op::SgInject { .. }))
        });

        let buffer_layout = mega_kernel_abi::BufferLayout::from_graph_geometry(
            &geometry, config.max_seq_len, sg_enabled,
        );

        // JIT pipeline
        let bottleneck_map = pain_point::PainPointAnalyzer::analyze(&graph, &self.profile);
        eprintln!("[JIT-TIME] PainPointAnalyzer: {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
        let t1 = std::time::Instant::now();
        if std::env::var("GLLM_DEBUG_RESOURCE").is_ok() {
            eprintln!("[R0/from_graph] PainPointAnalyzer: {} GEMMs analyzed, ridge={:.1}",
                bottleneck_map.gemm_bottlenecks.len(), bottleneck_map.ridge_point);
        }
        let exec_plan = planner::ExecutionPlan::from_profile_with_bottlenecks(&self.profile, bottleneck_map);
        let bottleneck_map = exec_plan.op_bottleneck_map.as_ref().unwrap();

        eprintln!("[JIT-TIME] ExecutionPlan+SemanticDAG+Fusion+HW: {:.2}ms", t1.elapsed().as_secs_f64() * 1000.0);
        let t2 = std::time::Instant::now();

        let registry = ScalarOpRegistry::with_defaults();
        let semantic_dag = SemanticDAG::from_graph(&graph, &registry);

        let mut fusion_plan = fusion::fuse_with_dag_prebuilt(&graph, &semantic_dag, &exec_plan, Some(bottleneck_map), None);
        hw_constraints::enforce_constraints(&mut fusion_plan.groups, &graph, &exec_plan);

        // REQ-UMK-31: Analyze parallel scheduling plan for Phase 2/3.
        // Same-level fusion groups in TopoLevel have zero data dependency and can
        // be emitted/lowered in parallel. The scheduler decides parallelism degree.
        let mut parallel_scheduler = parallel_compile::ParallelCompileScheduler::new();
        parallel_scheduler.analyze_schedule(&fusion_plan, &graph);
        if std::env::var("GLLM_DEBUG_RESOURCE").is_ok() {
            eprintln!("[PARALLEL-SCHED] {}", parallel_scheduler.summary());
        }

        let accel_registry = accel_registry::AccelerationRegistry::new();
        let layout_assignment = layout_negotiator::LayoutNegotiator::negotiate(
            &fusion_plan.groups, &accel_registry, &self.profile, &semantic_dag, bottleneck_map, &graph,
        );

        let virtual_tensor_map = virtual_tensor::DataFlowOptimizer::eliminate(
            &graph, &fusion_plan, Some(&layout_assignment), &self.profile,
        );
        eprintln!("[JIT-TIME] VTM+LayoutNegotiate: {:.2}ms", t2.elapsed().as_secs_f64() * 1000.0);
        let t3 = std::time::Instant::now();
        if std::env::var("GLLM_DEBUG_RESOURCE").is_ok() {
            eprintln!("[VTM-mega] virtualized {} tensors, bytes_saved={}",
                virtual_tensor_map.virtual_map.len(),
                virtual_tensor_map.bytes_saved);
            for (&tid, vt) in &virtual_tensor_map.virtual_map {
                let name = graph.tensor(tid).map(|t| t.name.as_str()).unwrap_or("?");
                let src_name = graph.tensor(vt.source).map(|t| t.name.as_str()).unwrap_or("?");
                eprintln!("[VTM-mega]   {} ({:?}) -> virtual from {} ({:?})",
                    name, tid, src_name, vt.source);
            }
        }

        let virtual_activation = virtual_activation::VirtualActivationMap::analyze(&graph, &fusion_plan);

        let lifetimes = buffer_alloc::analyze_lifetimes(
            &graph, &fusion_plan, Some(&virtual_tensor_map), Some(&virtual_activation),
        );
        let alloc = buffer_alloc::allocate_buffers_aligned(
            &lifetimes, self.profile.hw_info.cacheline_bytes,
            Some(&virtual_tensor_map), Some(&virtual_activation), &graph,
            Some(&layout_assignment),
        );

        #[cfg(feature = "jit-x86")]
        {
            use codegen::vm::mega_kernel_emit::compile_mega_kernel_vm;
            use codegen::vm::{isa_profile::IsaProfile, isa_hook, reg_alloc::RegAllocator,
                              stack_frame::StackFrame, x86_lower::X86Lower,
                              opt_pass::PassRegistry};
            use codegen::vm::resource_planner::plan_mega_kernel_resources;

            let profile = IsaProfile::from_device_profile(&self.profile);
            let hook = isa_hook::select_hook(&profile);
            let hook_ref: Option<&dyn isa_hook::IsaHook> = Some(&*hook);

            let elem_bytes = geometry.compute_dtype.size_bytes();
            let activation_bytes = config.max_seq_len * geometry.hidden * elem_bytes;
            // [LEGAL-PSC28] elem_bytes comes from geometry.compute_dtype.size_bytes() (dtype-inferred),
            // not hardcoded F32. compute_dtype is derived from (storage_dtype, DeviceProfile) per
            // REQ-DTYPE-CHAIN-005. This satisfies ARCH-DTYPE-JIT-TYPED.
            // [FIX-PSC27] kv_bytes uses KV-specific dimensions (num_kv_heads * head_dim) instead of
            // geometry.hidden. For MHA models: num_kv_heads == num_heads, so kv_dim == hidden (no change).
            // For GQA/MQA models: num_kv_heads < num_heads, so kv_dim < hidden (saves memory).
            // The *2 accounts for K and V halves of the KV cache.
            let kv_dim = geometry.num_kv_heads * geometry.head_dim;
            let kv_bytes = config.max_seq_len * kv_dim * 2;
            let resource_plan = plan_mega_kernel_resources(
                &graph, &fusion_plan, &profile, &alloc,
                geometry.hidden, activation_bytes, kv_bytes,
            );
            if std::env::var("GLLM_DEBUG_RESOURCE").is_ok() {
                eprintln!("[GRP] Resource plan: {} groups, scratchpad={} bytes, stack={} bytes, peak_vec={}, peak_gpr={}",
                    resource_plan.summary.num_layers,
                    resource_plan.summary.total_scratchpad_bytes,
                    resource_plan.summary.total_stack_bytes,
                    resource_plan.summary.peak_vec_regs,
                    resource_plan.summary.peak_gpr_regs);
            }

            let topology = codegen::vm::topology::GraphTopologyAnalysis::analyze(
                &graph,
            );
            // Extract output_float_elems before topology is moved into compile_mega_kernel_vm.
            // For SinglePass (non-generate) graphs, we need to copy output from scratchpad
            // back to the ABI output arg. The output tensor is determined by:
            // 1. topology.logits_output_tid (if Argmax is present → its input tensor)
            // 2. graph.outputs[0] (fallback: no Argmax, e.g. GEMM/embedding/reranker)
            let output_float_elems = if matches!(topology.loop_topology, crate::compiler::codegen::vm::topology::LoopTopology::SinglePass) {
                let output_tid = topology.logits_output_tid
                    .or_else(|| graph.outputs.first().copied());
                output_tid
                    .and_then(|tid| graph.tensor(tid))
                    .map(|t| t.shape.iter().map(|d| match d {
                        SymDim::Concrete(v) => *v,
                        SymDim::Symbolic { max_value: Some(m), .. } => *m,
                        _ => 1,
                    }).product::<usize>())
                    .unwrap_or(0)
            } else {
                0
            };
            let t4 = std::time::Instant::now();
            let (mut program, rope_cache, logits_scratch_offset) = compile_mega_kernel_vm(
                &fusion_plan, &graph, &alloc, Some(&registry), &profile,
                hook_ref, &buffer_layout, Some(bottleneck_map), Some(&virtual_activation),
                Some(&virtual_tensor_map), Some(&layout_assignment),
                config.debug_jit,
                None, // mtp_config now derived from topology.mtp_config (SPEC/39)
                Some(&resource_plan),
                topology,
            ).map_err(InferenceError::CompileError)?;
            eprintln!("[JIT-TIME] compile_mega_kernel_vm: {:.2}ms ({} VmInstrs)", t4.elapsed().as_secs_f64() * 1000.0, program.len());
            // Dump VmProgram for debugging
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create("/tmp/gllm_vmprogram_dump.txt") {
                    for (i, instr) in program.instrs.iter().enumerate() {
                        writeln!(f, "{:5}: {:?}", i, instr).ok();
                    }
                }
            }

            let pass_registry = PassRegistry::with_defaults();
            pass_registry.run_all(&mut program, &profile, &*hook);

            let t5 = std::time::Instant::now();
            let alloc_result = RegAllocator::new(&profile).allocate(&program)
                .map_err(|e| InferenceError::CompileError(format!("RegAlloc: {}", e).into()))?;
            eprintln!("[JIT-TIME] RegAllocator: {:.2}ms", t5.elapsed().as_secs_f64() * 1000.0);
            // Dump reg alloc for debugging
            {
                use std::io::Write;
                let mut f = std::fs::File::create(format!("/tmp/gllm_regalloc_dump_{}.txt", compile_id)).unwrap();
                for spill in &alloc_result.spills {
                    writeln!(f, "SPILL v{} offset={} size={}", spill.vreg.0, spill.offset, spill.size).unwrap();
                }
                let mut regs: Vec<_> = alloc_result.mapping.iter().collect();
                regs.sort_by_key(|(v, _)| v.0);
                for (v, p) in &regs {
                    writeln!(f, "MAP v{} → {:?}", v.0, p).unwrap();
                }
            }

            let frame = StackFrame::compute(&alloc_result, &profile, 0);

            use codegen::vm::plan_lower::SymDimSlotMap;
            let sym_slot_map = SymDimSlotMap::mega_kernel_abi();
            let has_avx512 = match &profile.platform {
                codegen::vm::isa_profile::Platform::X86_64 { has_avx512, .. } => *has_avx512,
                _ => false,
            };
            let mut lowerer = X86Lower::with_sym_map(has_avx512, sym_slot_map);
            lowerer.set_scratch_gprs(&profile.scratch_gprs)
                .map_err(InferenceError::CompileError)?;
            lowerer.set_scratch_vec_regs(&profile.scratch_vec_regs)
                .map_err(InferenceError::CompileError)?;
            lowerer.precompute_zero_vregs(&program);
            lowerer.emit_prologue(&frame, &alloc_result)
                .map_err(InferenceError::CompileError)?;
            // StackLayout 现在在 emit_prologue 内部直接构建，无需 set_spill_base
            let t6 = std::time::Instant::now();
            for instr in &program.instrs {
                lowerer.lower_instr(instr, &alloc_result)
                    .map_err(InferenceError::CompileError)?;
            }
            eprintln!("[JIT-TIME] X86Lower ({} instrs): {:.2}ms", program.len(), t6.elapsed().as_secs_f64() * 1000.0);
            lowerer.emit_epilogue(&frame, &alloc_result)
                .map_err(InferenceError::CompileError)?;
            let lowerer_source_map = lowerer.take_source_map();
            let code = lowerer.finalize()
                .map_err(InferenceError::CompileError)?;
            eprintln!("[JIT-TIME] Total code size: {} bytes", code.len());
            // Dump JIT machine code for debugging
            {
                if let Ok(mut f) = std::fs::File::create("/tmp/gllm_jit_code.bin") {
                    use std::io::Write;
                    f.write_all(&code).ok();
                }
            }

            let elem_bytes = geometry.compute_dtype.size_bytes();
            let vocab_bytes = geometry.vocab_size * elem_bytes;
            /// Sampling workspace: 4x vocab_bytes (indices + PRNG + reserved CDF + reserved temp).
            /// See gllm abi_types.inc.rs SAMPLING_WORKSPACE_MULTIPLIER for LEGAL justification.
            const SAMPLING_WORKSPACE_MULTIPLIER: usize = 4;
            let sampling_bytes = vocab_bytes * SAMPLING_WORKSPACE_MULTIPLIER;
            // BCE-20260623-001 fix: logits_end must cover both generate-mode (vocab_size)
            // and single-pass mode (output_float_elems). For non-generate graphs (no Argmax),
            // output tensor is written to scratchpad[logits_scratch_offset] and
            // copy_nonoverlapping reads output_float_elems f32 from there — the scratchpad
            // must be large enough to hold the full output tensor.
            let generate_logits_bytes = config.max_seq_len * geometry.vocab_size * elem_bytes;
            let single_pass_output_bytes = output_float_elems * elem_bytes;
            let logits_end = logits_scratch_offset + generate_logits_bytes.max(single_pass_output_bytes);
            let sg_end = if buffer_layout.sg_data_bytes > 0 {
                let sg_start = (logits_scratch_offset + vocab_bytes + sampling_bytes + 63) & !63;
                sg_start + geometry.hidden * elem_bytes * 2
            } else {
                0
            };
            // DWC padded buffer must fit within scratchpad (compute_dwc_requirement mirrors mega_kernel_emit).
            let dwc_end = match codegen::vm::plan_lower::compute_dwc_requirement(
                &fusion_plan, &graph, &alloc, rope_cache.as_ref(), None,
            ) {
                Ok(Some(req)) => req.padded_offset + req.total_bytes,
                _ => 0,
            };
            let total_scratch = logits_end
                .max(buffer_layout.total_scratchpad_bytes)
                .max(sg_end)
                .max(dwc_end)
                .max(64);

            let hash = self.graph_content_hash(&graph);
            let mut layer = CompiledLayer::from_code(&code, total_scratch, hash)?;
            layer.weight_layout = Some(graph.weight_layout());
            layer.logits_scratch_offset = logits_scratch_offset;
            layer.output_float_elems = output_float_elems;

            let source_map = if config.debug_jit {
                let mut map = lowerer_source_map;
                map.sort_by_offset();
                Some(map)
            } else {
                None
            };

            // BCE-20260623-001 regression guard: scratchpad must be large enough
            // for the output copy in execute_as_mega_kernel (copy_nonoverlapping
            // reads output_float_elems f32 from scratchpad[logits_scratch_offset]).
            debug_assert!(
                total_scratch >= logits_scratch_offset + output_float_elems * elem_bytes,
                "scratchpad too small for output: total={} need offset={} + elems={} * {} = {}",
                total_scratch, logits_scratch_offset, output_float_elems, elem_bytes,
                logits_scratch_offset + output_float_elems * elem_bytes,
            );

            Ok(MegaKernelCompileOutput {
                layer_code: layer,
                weight_layout: mega_kernel_abi::KernelWeightLayout::from_graph_geometry(&geometry),
                buffer_layout,
                num_layers: geometry.num_layers,
                vocab_size: geometry.vocab_size,
                hidden: geometry.hidden,
                rope_cache,
                total_scratchpad_bytes: total_scratch,
                logits_scratch_offset,
                hetero_layout,
                source_map,
                tensor_sources: {
                    let mut ts = alloc.tensor_sources.clone();
                    for op in &graph.ops {
                        if let crate::compiler::graph::Op::Gather { .. } | crate::compiler::graph::Op::QuantGather { .. } = &op.op {
                            if let Some(&out_tid) = op.outputs.first() {
                                let off = alloc.offset_of(out_tid).unwrap_or(0);
                                ts.insert(out_tid, crate::compiler::buffer_alloc::TensorPtrSource::Intermediate { offset: off });
                            }
                        }
                    }
                    ts
                },
            })
        }

        #[cfg(not(feature = "jit-x86"))]
        {
            let _ = (fusion_plan, alloc);
            Err(InferenceError::CompileError("JIT backend not enabled (feature jit-x86 required)".into()))
        }
    }

    /// GPU codegen path — emits PTX/HIP/MSL source.
    ///
    /// Internal helper called by `compile()` when `config.target == Gpu`.
    /// Same Scalar + SymExec → Fusion pipeline as CPU, but uses GpuLower for ISA Lowering.
    #[cfg(any(feature = "jit-cuda", feature = "jit-hip"))]
    fn compile_gpu(
        &mut self,
        graph: CompilerGraph,
        config: &mega_kernel_abi::CompileConfig,
        sm_version: u32,
    ) -> Result<GpuMegaKernelOutput, InferenceError> {
        use codegen::vm::gpu_lower::{GpuLower, GpuDialect};
        use codegen::vm::reg_alloc::RegAllocator;
        use codegen::vm::stack_frame::StackFrame;
        use codegen::vm::isa_profile::IsaProfile;
        use codegen::vm::opt_pass::PassRegistry;

        let geometry = graph_geometry::GraphDerivedGeometry::from_graph(&graph, &DeviceProfile::detect())
            .map_err(|e| InferenceError::CompileError(format!("GraphDerivedGeometry: {}", e).into()))?;

        // [FIX-PSC29] Determine SG presence from graph ops (SgDetect/SgInject).
        // When no SG ops exist, BufferLayout skips SG allocation entirely.
        let sg_enabled = graph.ops.iter().any(|op| {
            matches!(op.op_resolved(&graph), Some(graph::Op::SgDetect { .. }) | Some(graph::Op::SgInject { .. }))
        });

        let buffer_layout = mega_kernel_abi::BufferLayout::from_graph_geometry(
            &geometry, config.max_seq_len, sg_enabled,
        );

        let bottleneck_map = pain_point::PainPointAnalyzer::analyze(&graph, &self.profile);
        let exec_plan = planner::ExecutionPlan::from_profile_with_bottlenecks(&self.profile, bottleneck_map);
        let bottleneck_map = exec_plan.op_bottleneck_map.as_ref().unwrap();

        let registry = ScalarOpRegistry::with_defaults();
        let semantic_dag = SemanticDAG::from_graph(&graph, &registry);
        let mut fusion_plan = fusion::fuse_with_dag_prebuilt(&graph, &semantic_dag, &exec_plan, Some(bottleneck_map), None);
        hw_constraints::enforce_constraints(&mut fusion_plan.groups, &graph, &exec_plan);

        let accel_registry = accel_registry::AccelerationRegistry::new();
        let layout_assignment = layout_negotiator::LayoutNegotiator::negotiate(
            &fusion_plan.groups, &accel_registry, &self.profile, &semantic_dag, bottleneck_map, &graph,
        );
        let virtual_tensor_map = virtual_tensor::DataFlowOptimizer::eliminate(
            &graph, &fusion_plan, Some(&layout_assignment), &self.profile,
        );
        let virtual_activation = virtual_activation::VirtualActivationMap::analyze(&graph, &fusion_plan);
        let lifetimes = buffer_alloc::analyze_lifetimes(
            &graph, &fusion_plan, Some(&virtual_tensor_map), Some(&virtual_activation),
        );
        let alloc = buffer_alloc::allocate_buffers_aligned(
            &lifetimes, self.profile.hw_info.cacheline_bytes,
            Some(&virtual_tensor_map), Some(&virtual_activation), &graph,
            Some(&layout_assignment),
        );

        let profile = IsaProfile::from_device_profile(&self.profile);
        let elem_bytes = geometry.compute_dtype.size_bytes();
        // [LEGAL-PSC28] elem_bytes comes from geometry.compute_dtype.size_bytes() (dtype-inferred),
        // not hardcoded F32. compute_dtype is derived from (storage_dtype, DeviceProfile) per
        // REQ-DTYPE-CHAIN-005. This satisfies ARCH-DTYPE-JIT-TYPED.
        let activation_bytes = config.max_seq_len * geometry.hidden * elem_bytes;
        // [FIX-PSC27] kv_bytes uses KV-specific dimensions (num_kv_heads * head_dim) instead of
        // geometry.hidden. For MHA models: num_kv_heads == num_heads, so kv_dim == hidden (no change).
        // For GQA/MQA models: num_kv_heads < num_heads, so kv_dim < hidden (saves memory).
        // The *2 accounts for K and V halves of the KV cache.
        let kv_dim = geometry.num_kv_heads * geometry.head_dim;
        let kv_bytes = config.max_seq_len * kv_dim * 2;
        let resource_plan = codegen::vm::resource_planner::plan_mega_kernel_resources(
            &graph, &fusion_plan, &profile, &alloc,
            geometry.hidden, activation_bytes, kv_bytes,
        );
        let topology = codegen::vm::topology::GraphTopologyAnalysis::analyze(
            &graph,
        );
        // BCE-20260623-001: Extract output_float_elems before topology is moved.
        let gpu_output_float_elems = if matches!(topology.loop_topology, crate::compiler::codegen::vm::topology::LoopTopology::SinglePass) {
            let output_tid = topology.logits_output_tid
                .or_else(|| graph.outputs.first().copied());
            output_tid
                .and_then(|tid| graph.tensor(tid))
                .map(|t| t.shape.iter().map(|d| match d {
                    SymDim::Concrete(v) => *v,
                    SymDim::Symbolic { max_value: Some(m), .. } => *m,
                    _ => 1,
                }).product::<usize>())
                .unwrap_or(0)
        } else {
            0
        };
        let (mut program, rope_cache, logits_scratch_offset) =
            codegen::vm::mega_kernel_emit::compile_mega_kernel_vm(
                &fusion_plan, &graph, &alloc, Some(&registry), &profile,
                None, &buffer_layout, Some(bottleneck_map),
                Some(&virtual_activation), Some(&virtual_tensor_map), Some(&layout_assignment),
                false, None, Some(&resource_plan),
                topology,
            ).map_err(|e| InferenceError::CompileError(e.into()))?;

        let pass_registry = PassRegistry::with_defaults();
        let hook = codegen::vm::isa_hook::select_hook(&profile);
        pass_registry.run_all(&mut program, &profile, &*hook);

        let alloc_result = RegAllocator::new(&profile).allocate(&program)
            .map_err(|e| InferenceError::CompileError(format!("RegAlloc: {}", e).into()))?;
        let frame = StackFrame::compute(&alloc_result, &profile, 0);

        let dialect = if cfg!(feature = "jit-cuda") {
            GpuDialect::Ptx { sm_version }
        } else {
            GpuDialect::Hip { gfx_arch: 942, wave_size: 64 } // MI300 default
        };
        let mut lowerer = GpuLower::new(dialect);
        let vreg_counts = program.vreg_counts_by_kind();
        lowerer.emit_mega_kernel_prologue(&frame, &alloc_result, vreg_counts)
            .map_err(|e| InferenceError::CompileError(e.into()))?;
        lowerer.set_vreg_kind_map(&program);

        for instr in &program.instrs {
            lowerer.lower_instr(instr, &alloc_result)
                .map_err(|e| InferenceError::CompileError(e.into()))?;
        }
        lowerer.emit_epilogue(&frame, &alloc_result)
            .map_err(|e| InferenceError::CompileError(e.into()))?;
        let gpu_code = lowerer.finalize()
            .map_err(|e| InferenceError::CompileError(e.into()))?;

        let elem_bytes = geometry.compute_dtype.size_bytes();
        let vocab_bytes = geometry.vocab_size * elem_bytes;
        /// Sampling workspace: 4x vocab_bytes (indices + PRNG + reserved CDF + reserved temp).
        /// See gllm abi_types.inc.rs SAMPLING_WORKSPACE_MULTIPLIER for LEGAL justification.
        const SAMPLING_WORKSPACE_MULTIPLIER: usize = 4;
        let sampling_bytes = vocab_bytes * SAMPLING_WORKSPACE_MULTIPLIER;
        // BCE-20260623-001 fix (GPU path): total_scratch must also cover
        // single-pass output tensor bytes when output_float_elems > 0.
        let generate_logits_bytes = config.max_seq_len * geometry.vocab_size * elem_bytes;
        let single_pass_output_bytes = gpu_output_float_elems * elem_bytes;
        let logits_end = logits_scratch_offset + generate_logits_bytes.max(single_pass_output_bytes);
        // [FIX-PSC5] GPU path must include sg_end and dwc_end in scratchpad sizing,
        // mirroring the CPU path (see compile_cpu sg_end/dwc_end calculation).
        // Without this, models with SG/DWC ops get insufficient scratchpad → OOB access.
        let sg_end = if buffer_layout.sg_data_bytes > 0 {
            let sg_start = (logits_scratch_offset + vocab_bytes + sampling_bytes + 63) & !63;
            sg_start + geometry.hidden * elem_bytes * 2
        } else {
            0
        };
        // DWC padded buffer must fit within scratchpad (compute_dwc_requirement mirrors mega_kernel_emit).
        let dwc_end = match codegen::vm::plan_lower::compute_dwc_requirement(
            &fusion_plan, &graph, &alloc, rope_cache.as_ref(), None,
        ) {
            Ok(Some(req)) => req.padded_offset + req.total_bytes,
            _ => 0,
        };
        let total_scratch = logits_end
            .max(buffer_layout.total_scratchpad_bytes)
            .max(sg_end)
            .max(dwc_end)
            .max(64);

        Ok(GpuMegaKernelOutput {
            gpu_code: gpu_code.into_bytes(),
            total_scratchpad_bytes: total_scratch,
            num_layers: geometry.num_layers,
            vocab_size: geometry.vocab_size,
            hidden: geometry.hidden,
            rope_cache,
            logits_scratch_offset,
        })
    }

    /// Compile a forward-only CompilerGraph to GPU PTX/HIP/MSL code.

    /// Compute a deterministic content hash for a CompilerGraph.
    ///
    /// Hash inputs: op kinds in topological order, edge connections (tensor IDs),
    /// tensor shapes, and hardware fingerprint. Uses the standard library's
    /// `DefaultHasher` (SipHash-1-3) for collision resistance.
    fn graph_content_hash(&self, graph: &CompilerGraph) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Phase 8: OPCODE_VERSION — Op enum 结构变更时 bump，自动失效旧 cache。
        crate::compiler::graph::OPCODE_VERSION.hash(&mut hasher);

        // Ops in topological order for determinism
        let topo = graph.topological_sort();
        topo.len().hash(&mut hasher);

        for &op_id in &topo {
            if let Some(op) = graph.op(op_id) {
                // Op 内容指纹（胖 opcode 自描述，单 IR）
                op.op.content_hash(&mut hasher);
                // Edge connections
                for &tid in &op.inputs {
                    tid.0.hash(&mut hasher);
                }
                for &tid in &op.outputs {
                    tid.0.hash(&mut hasher);
                }
            }
        }

        // Tensor shapes and dtypes
        for t in &graph.tensors {
            t.id.0.hash(&mut hasher);
            t.shape.hash(&mut hasher);
            t.dtype.size_bytes().hash(&mut hasher);
        }

        // Hardware fingerprint — same graph on different HW produces different code
        self.profile.hw_info.fingerprint().hash(&mut hasher);

        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{Op, GemmSpec, NormSpec, QuantGemmSpec};
    use crate::types::{ModelConfig, DType};

    #[test]
    fn test_compiler_new() {
        let compiler = InferenceCompiler::new();
        assert_eq!(compiler.cache_size(), 0);
    }

    /// End-to-end: full JIT pipeline for LLaMA-7B decoder layer.
    #[test]
    fn test_e2e_llama_7b() {
        use crate::compiler::planner::ExecutionPlan;
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        // Build CompilerGraph DAG
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        assert!(graph.num_ops() >= 14);

        // Fusion
        let registry = registry::ScalarOpRegistry::with_defaults();
        let fplan = fusion::fuse_with_dag(&graph, &registry, &exec_plan);
        assert!(fplan.num_groups() < graph.num_ops());

        // ISA Lowering (Codegen)
        #[cfg(feature = "jit-x86")]
        {
            let dag = semantic_dag::SemanticDAG::from_graph(&graph, &registry);
            let fusion_plan = fusion::fuse_with_dag_prebuilt(&graph, &dag, &exec_plan, None, None);
            let lifetimes = buffer_alloc::analyze_lifetimes(&graph, &fusion_plan, None, None);
            let alloc = buffer_alloc::allocate_buffers(&lifetimes);
            let mut cg = codegen::X86CodeGen::new(&profile, graph.infer_computation_dtype());
            let output = cg.emit_plan(&fusion_plan, &graph, &alloc, &exec_plan, Some(&registry))
                .expect("JIT codegen failed");
            let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap();
            assert!(layer.code_size() > 0);
            eprintln!(
                "E2E LLaMA-7B: {} ops → {} groups → {} bytes code, {} bytes scratch",
                graph.num_ops(),
                fplan.num_groups(),
                layer.code_size(),
                output.scratchpad_bytes,
            );
        }
        #[cfg(not(feature = "jit-x86"))]
        {
            eprintln!(
                "E2E LLaMA-7B: {} ops → {} groups (Phase 3 skipped, jit-x86 not enabled)",
                graph.num_ops(),
                fplan.num_groups(),
            );
        }
    }

    /// End-to-end: full JIT pipeline for Gemma-2B (GeGLU variant).
    #[test]
    fn test_e2e_gemma_2b() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let registry = registry::ScalarOpRegistry::with_defaults();
        let fplan = fusion::fuse_with_dag(&graph, &registry, &exec_plan);

        #[cfg(feature = "jit-x86")]
        {
            let mk_config = mega_kernel_abi::CompileConfig {
                max_seq_len: 512,
                debug_jit: false,
                hetero: None,
                target: mega_kernel_abi::CompileTarget::Cpu,
            };
            let mut compiler = InferenceCompiler::with_profile(profile);
            let output = compiler.compile(graph.clone(), &mk_config, None)
                .expect("compile (Cpu) failed")
                .expect_cpu();
            assert!(output.layer_code.code_size() > 0);
            eprintln!(
                "E2E Gemma-2B: {} ops → {} groups → {} bytes code",
                graph.num_ops(),
                fplan.num_groups(),
                output.layer_code.code_size(),
            );
        }
        #[cfg(not(feature = "jit-x86"))]
        {
            eprintln!(
                "E2E Gemma-2B: {} ops → {} groups (Phase 3 skipped, jit-x86 not enabled)",
                graph.num_ops(),
                fplan.num_groups(),
            );
        }
    }

    #[test]
    fn test_graph_content_hash_deterministic() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        let compiler = InferenceCompiler::with_profile(profile);
        let h1 = compiler.graph_content_hash(&graph);
        let h2 = compiler.graph_content_hash(&graph);
        assert_eq!(h1, h2, "same graph must produce identical hash");
        assert_ne!(h1, 0, "hash should not be zero");
    }

    #[test]
    fn test_graph_content_hash_differs_for_different_graphs() {
        let profile = DeviceProfile::detect();
        let compiler = InferenceCompiler::with_profile(profile.clone());

        let config_a = ModelConfig::llama_7b();
        let ir_a = LayerIR::from_model_config(&config_a, 1);
        let graph_a =
            CompilerGraph::from_layer_ir(&ir_a, &profile).expect("from_layer_ir failed");

        let config_b = ModelConfig::gemma_2b();
        let ir_b = LayerIR::from_model_config(&config_b, 1);
        let graph_b =
            CompilerGraph::from_layer_ir(&ir_b, &profile).expect("from_layer_ir failed");

        let ha = compiler.graph_content_hash(&graph_a);
        let hb = compiler.graph_content_hash(&graph_b);
        assert_ne!(ha, hb, "different graphs should produce different hashes");
    }

    #[test]
    fn test_compile_graph_nonzero_hash() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        let mk_config = mega_kernel_abi::CompileConfig {
            max_seq_len: 512,
            debug_jit: false,
            hetero: None,
            target: mega_kernel_abi::CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::with_profile(profile);
        let output = compiler.compile(graph, &mk_config, None)
            .expect("compile (Cpu) failed")
            .expect_cpu();
        assert_ne!(output.layer_code.config_hash, 0, "compile should produce a non-zero hash");
    }

    // ── 13 new tests (+9 existing = 22 total) ──────────────────────────

    /// Verify MegaKernelCompileOutput can be constructed with struct
    /// literal syntax and that its fields hold the expected values.
    /// Tests struct constructor + field boundary values without running
    /// the full JIT pipeline.
    #[test]
    fn test_mega_kernel_compile_output_struct_constructor() {
        // Arrange: build a minimal CompiledLayer via compile (Cpu target).
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let mk_config = mega_kernel_abi::CompileConfig {
            max_seq_len: 512,
            debug_jit: false,
            hetero: None,
            target: mega_kernel_abi::CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::with_profile(profile);
        let compiled = compiler.compile(graph, &mk_config, None)
            .expect("compile (Cpu) failed")
            .expect_cpu();
        let layer = compiled.layer_code;

        // Act: construct MegaKernelCompileOutput manually with explicit layouts
        let zero_per_layer = mega_kernel_abi::PerLayerWeightLayout {
            attn_norm_offset: 0, attn_norm_bytes: 0,
            w_q_offset: 0, w_q_bytes: 0,
            w_k_offset: 0, w_k_bytes: 0,
            w_v_offset: 0, w_v_bytes: 0,
            w_o_offset: 0, w_o_bytes: 0,
            w_q_norm_offset: 0, w_q_norm_bytes: 0,
            w_k_norm_offset: 0, w_k_norm_bytes: 0,
            ffn_norm_offset: 0, ffn_norm_bytes: 0,
            w_gate_offset: 0, w_gate_bytes: 0,
            w_up_offset: 0, w_up_bytes: 0,
            w_down_offset: 0, w_down_bytes: 0,
        };
        let weight_layout = mega_kernel_abi::KernelWeightLayout {
            embed_offset: 0, embed_bytes: 0,
            layer_0_offset: 0, layer_stride: 0,
            per_layer: zero_per_layer,
            final_norm_offset: 0, final_norm_bytes: 0,
            logits_producer_offset: 0, logits_producer_bytes: 0,
            total_bytes: 0,
        };
        let buffer_layout = mega_kernel_abi::BufferLayout {
            activation_a_offset: 0, activation_b_offset: 0,
            activation_bytes: 0,
            logits_offset: 0, logits_bytes: 0,
            sampling_workspace_offset: 0, sampling_workspace_bytes: 0,
            sg_detect_offset: 0, sg_knowledge_offset: 0,
            sg_data_bytes: 0,
            total_scratchpad_bytes: 0,
        };
        let output = MegaKernelCompileOutput {
            layer_code: layer,
            weight_layout,
            buffer_layout,
            num_layers: 32,
            vocab_size: 32000,
            hidden: 4096,
            rope_cache: None,
            total_scratchpad_bytes: 1024,
            logits_scratch_offset: 512,
            hetero_layout: None,
            tensor_sources: std::collections::HashMap::new(),
            source_map: None,
        };

        // Assert: field values preserved
        assert_eq!(output.num_layers, 32);
        assert_eq!(output.vocab_size, 32000);
        assert_eq!(output.hidden, 4096);
        assert_eq!(output.total_scratchpad_bytes, 1024);
        assert_eq!(output.logits_scratch_offset, 512);
        assert!(output.rope_cache.is_none());
        assert!(output.hetero_layout.is_none());
        assert!(output.source_map.is_none());
        assert!(output.layer_code.code_size() > 0);
    }

    /// Verify GpuMegaKernelOutput Debug derive produces readable output
    /// and that all-None rope_cache is properly held.
    #[test]
    fn test_gpu_mega_kernel_output_debug_format() {
        let output = GpuMegaKernelOutput {
            gpu_code: vec![0x90, 0x90, 0x90], // NOP NOP NOP
            total_scratchpad_bytes: 4096,
            num_layers: 4,
            vocab_size: 32000,
            hidden: 4096,
            rope_cache: None,
            logits_scratch_offset: 2048,
        };

        // Act: Debug format
        let debug_str = format!("{:?}", output);

        // Assert: key fields appear in debug output
        assert!(debug_str.contains("gpu_code"));
        assert!(debug_str.contains("total_scratchpad_bytes"));
        assert!(debug_str.contains("32000"));
        assert!(debug_str.contains("None"));
        assert_eq!(output.gpu_code.len(), 3);
        assert_eq!(output.num_layers, 4);
    }

    /// Verify GpuForwardOutput Debug derive and basic field access.
    #[test]
    fn test_gpu_forward_output_fields() {
        let output = GpuForwardOutput {
            gpu_code: Vec::new(),
            total_scratchpad_bytes: 0,
        };

        // Act & Assert: empty GPU code, zero scratchpad
        assert!(output.gpu_code.is_empty());
        assert_eq!(output.total_scratchpad_bytes, 0);

        // Assert: Debug formats without panic
        let _ = format!("{:?}", output);
    }

    /// Verify CompileGuard RAII: acquiring the lock succeeds and dropping
    /// releases it so a second acquisition does not deadlock.
    #[test]
    fn test_compile_guard_raii_lock_release() {
        // Act: acquire and immediately drop
        {
            let _guard = compile_lock();
            // Guard held — we cannot test re-entrant lock here without
            // deadlock risk, but we can verify it does not panic.
        }

        // Assert: second acquisition succeeds after drop (no deadlock)
        let _guard2 = compile_lock();
        // If we reach here, the lock was properly released.
    }

    /// Verify StrategyBias default values are all 1.0 (neutral) except
    /// expert_eviction_aggressiveness (0.0).
    #[test]
    fn test_strategy_bias_default_values() {
        let bias = planner::StrategyBias::default();

        // Assert: all scaling fields default to 1.0
        assert_eq!(bias.fusion_cost_scale, 1.0);
        assert_eq!(bias.pipeline_cost_scale, 1.0);
        assert_eq!(bias.parallelism_cost_scale, 1.0);
        assert_eq!(bias.epilogue_depth_preference, 1.0);
        assert_eq!(bias.k_depth_preference, 1.0);
        assert_eq!(bias.kv_cache_budget_scale, 1.0);
        assert_eq!(bias.weight_prefetch_budget_scale, 1.0);
        assert_eq!(bias.batch_flexibility, 1.0);
        assert_eq!(bias.decode_ratio_scale, 1.0);
        assert_eq!(bias.speculative_decoding_value, 1.0);
        assert_eq!(bias.quantization_aggressiveness, 1.0);
        assert_eq!(bias.expert_prefetch_priority, 1.0);

        // Assert: eviction defaults to 0.0 (full resident)
        assert_eq!(bias.expert_eviction_aggressiveness, 0.0);
    }

    /// Verify StrategyBias::validate() clamps out-of-range values to
    /// valid bounds (not panic, not silently accept).
    #[test]
    fn test_strategy_bias_validate_clamps_extremes() {
        let mut bias = planner::StrategyBias {
            fusion_cost_scale: 0.0,
            batch_flexibility: 5.0,
            expert_eviction_aggressiveness: 10.0,
            expert_prefetch_priority: 0.0,
            ..planner::StrategyBias::default()
        };

        // Act
        bias.validate();

        // Assert: clamped to minimum bounds
        assert!(
            bias.fusion_cost_scale >= 0.2,
            "fusion_cost_scale clamped to min: got {}",
            bias.fusion_cost_scale,
        );
        assert!(
            bias.batch_flexibility <= 1.0,
            "batch_flexibility clamped to max: got {}",
            bias.batch_flexibility,
        );
        assert!(
            bias.expert_eviction_aggressiveness <= 2.0,
            "expert_eviction_aggressiveness clamped to max: got {}",
            bias.expert_eviction_aggressiveness,
        );
        assert!(
            bias.expert_prefetch_priority >= 0.1,
            "expert_prefetch_priority clamped to min: got {}",
            bias.expert_prefetch_priority,
        );
    }

    /// Verify GemmShape constructor, equality, and hash consistency.
    #[test]
    fn test_gemm_shape_equality_and_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = planner::GemmShape { m: 1, n: 4096, k: 4096 };
        let b = planner::GemmShape { m: 1, n: 4096, k: 4096 };
        let c = planner::GemmShape { m: 1, n: 4096, k: 2048 };

        // Assert: equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Assert: hash consistency
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish(), "equal shapes must have equal hashes");
    }

    /// Verify MicrokernelChoice fields and PartialEq behavior.
    #[test]
    fn test_microkernel_choice_variants() {
        let choice_a = planner::MicrokernelChoice { mr: 6, nr: 8 };
        let choice_b = planner::MicrokernelChoice { mr: 6, nr: 8 };
        let choice_c = planner::MicrokernelChoice { mr: 4, nr: 4 };

        // Assert: equality semantics
        assert_eq!(choice_a, choice_b);
        assert_ne!(choice_a, choice_c);

        // Assert: Debug formats without panic
        let _ = format!("{:?}", choice_a);
    }

    /// Verify BottleneckClass enum variants are distinct and Debug-able.
    #[test]
    fn test_bottleneck_class_variants() {
        let compute = planner::BottleneckClass::ComputeBound;
        let memory = planner::BottleneckClass::MemoryBound;
        let mixed = planner::BottleneckClass::Mixed;

        // Assert: all variants are distinct
        assert_ne!(compute, memory);
        assert_ne!(memory, mixed);
        assert_ne!(compute, mixed);

        // Assert: Debug output
        assert!(format!("{:?}", compute).contains("ComputeBound"));
        assert!(format!("{:?}", memory).contains("MemoryBound"));
        assert!(format!("{:?}", mixed).contains("Mixed"));
    }

    /// Verify FusionDecision enum variants are distinct and Clone works.
    #[test]
    fn test_fusion_decision_clone_and_equality() {
        let decisions = vec![
            planner::FusionDecision::RmsNormIntoGemm,
            planner::FusionDecision::GemmBiasAct(crate::traits::Activation::Silu),
            planner::FusionDecision::QkvSharedInput,
            planner::FusionDecision::FlashAttention,
            planner::FusionDecision::SwiGluFusion,
            planner::FusionDecision::GeGluFusion,
        ];

        // Assert: all variants are pairwise distinct
        for i in 0..decisions.len() {
            for j in (i + 1)..decisions.len() {
                assert_ne!(
                    decisions[i], decisions[j],
                    "FusionDecision variants at index {} and {} must differ",
                    i, j,
                );
            }
        }

        // Assert: Clone produces equal value
        let cloned = decisions[0].clone();
        assert_eq!(decisions[0], cloned);
    }

    /// Verify MoeConfig: fields, Clone, Copy, Debug all work correctly.
    #[test]
    fn test_moe_config_and_derives() {
        let moe = ir::MoeConfig { num_experts: 8, top_k: 2 };

        // Assert: fields
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);

        // Assert: Copy
        let moe_copy = moe;
        assert_eq!(moe_copy.num_experts, moe.num_experts);

        // Assert: Debug contains fields
        let moe_debug = format!("{:?}", moe);
        assert!(moe_debug.contains("num_experts"));
        assert!(moe_debug.contains("8"));
    }

    /// Verify CacheSource enum variants are distinct, Copy-able, and Debug-able.
    #[test]
    fn test_cache_source_variants() {
        let mem = cache::CacheSource::Memory;
        let disk = cache::CacheSource::Disk;

        // Assert: distinct
        assert_ne!(mem, disk);

        // Assert: Copy (assign from moved value)
        let mem_copy = mem;
        assert_eq!(mem, mem_copy);

        // Assert: Debug
        assert!(format!("{:?}", mem).contains("Memory"));
        assert!(format!("{:?}", disk).contains("Disk"));
    }

    /// Verify usize overflow safety in GemmPlan field arithmetic:
    /// mr * nr * nr_vecs should not overflow for realistic values.
    #[test]
    fn test_gemm_plan_struct_update_syntax_and_no_overflow() {
        let base = planner::GemmPlan {
            mr: 6,
            nr: 8,
            nr_vecs: 4,
            k_pipeline_depth: 3,
            pf_distance_a: 4096,
            pf_distance_b: 4096,
            max_epilogue_depth: 4,
            acc_regs: 16,
            scratch_regs: 8,
            strategy: planner::GemmMicrokernelStrategy::BlisAvx2,
        };

        // Act: struct update syntax to override one field
        let modified = planner::GemmPlan {
            mr: 4,
            ..base.clone()
        };

        // Assert: only overridden field changed
        assert_eq!(modified.mr, 4);
        assert_eq!(modified.nr, base.nr);
        assert_eq!(modified.strategy, base.strategy);

        // Assert: arithmetic on fields does not overflow
        let reg_product = base.mr.checked_mul(base.nr).expect("mr*nr overflow");
        let total_regs = reg_product.checked_mul(base.nr_vecs).expect("mr*nr*nr_vecs overflow");
        assert!(total_regs > 0, "register count product must be positive");
    }

    // ── 13 additional tests ──────────────────────────────────────────────

    #[test]
    fn test_inference_compiler_with_profile_empty_cache() {
        let profile = DeviceProfile::detect();
        let compiler = InferenceCompiler::with_profile(profile);
        assert_eq!(compiler.cache_size(), 0, "new compiler should have empty cache");
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let compiler = InferenceCompiler::with_profile(profile);

        let h1 = compiler.compute_hash(&ir);
        let h2 = compiler.compute_hash(&ir);
        assert_eq!(h1, h2, "same IR must produce identical hash");
        assert_ne!(h1, 0, "hash should not be zero");
    }

    #[test]
    fn test_compute_hash_differs_for_different_hidden() {
        let profile = DeviceProfile::detect();
        let compiler = InferenceCompiler::with_profile(profile.clone());

        let config_a = ModelConfig::llama_7b();
        let ir_a = LayerIR::from_model_config(&config_a, 1);

        let mut config_b = ModelConfig::llama_7b();
        config_b.hidden_size = 2048;
        let ir_b = LayerIR::from_model_config(&config_b, 1);

        let ha = compiler.compute_hash(&ir_a);
        let hb = compiler.compute_hash(&ir_b);
        assert_ne!(ha, hb, "different hidden_size should produce different hashes");
    }

    #[test]
    fn test_gpu_mega_kernel_output_field_access() {
        let output = GpuMegaKernelOutput {
            gpu_code: vec![0xAA, 0xBB],
            total_scratchpad_bytes: 8192,
            num_layers: 12,
            vocab_size: 50257,
            hidden: 768,
            rope_cache: None,
            logits_scratch_offset: 4096,
        };
        assert_eq!(output.gpu_code, vec![0xAA, 0xBB]);
        assert_eq!(output.total_scratchpad_bytes, 8192);
        assert_eq!(output.num_layers, 12);
        assert_eq!(output.vocab_size, 50257);
        assert_eq!(output.hidden, 768);
        assert!(output.rope_cache.is_none());
        assert_eq!(output.logits_scratch_offset, 4096);
    }

    #[test]
    fn test_gpu_forward_output_debug_format() {
        let output = GpuForwardOutput {
            gpu_code: vec![0x00],
            total_scratchpad_bytes: 1024,
        };
        let debug = format!("{:?}", output);
        assert!(debug.contains("gpu_code"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_compile_guard_drop_and_reacquire() {
        {
            let guard1 = compile_lock();
            drop(guard1);
        }
        // Must not deadlock
        let _guard2 = compile_lock();
    }

    #[test]
    fn test_mega_kernel_compile_output_field_access() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let mk_config = mega_kernel_abi::CompileConfig {
            max_seq_len: 512,
            debug_jit: false,
            hetero: None,
            target: mega_kernel_abi::CompileTarget::Cpu,
        };
        let mut compiler = InferenceCompiler::with_profile(profile);
        let compiled = compiler.compile(graph, &mk_config, None)
            .expect("compile (Cpu) failed")
            .expect_cpu();
        let layer = compiled.layer_code;

        let output = MegaKernelCompileOutput {
            layer_code: layer,
            weight_layout: mega_kernel_abi::KernelWeightLayout {
                embed_offset: 0, embed_bytes: 0,
                layer_0_offset: 0, layer_stride: 0,
                per_layer: mega_kernel_abi::PerLayerWeightLayout {
                    attn_norm_offset: 0, attn_norm_bytes: 0,
                    w_q_offset: 0, w_q_bytes: 0,
                    w_k_offset: 0, w_k_bytes: 0,
                    w_v_offset: 0, w_v_bytes: 0,
                    w_o_offset: 0, w_o_bytes: 0,
                    w_q_norm_offset: 0, w_q_norm_bytes: 0,
                    w_k_norm_offset: 0, w_k_norm_bytes: 0,
                    ffn_norm_offset: 0, ffn_norm_bytes: 0,
                    w_gate_offset: 0, w_gate_bytes: 0,
                    w_up_offset: 0, w_up_bytes: 0,
                    w_down_offset: 0, w_down_bytes: 0,
                },
                final_norm_offset: 0, final_norm_bytes: 0,
                logits_producer_offset: 0, logits_producer_bytes: 0,
                total_bytes: 0,
            },
            buffer_layout: mega_kernel_abi::BufferLayout {
                activation_a_offset: 0, activation_b_offset: 0,
                activation_bytes: 0,
                logits_offset: 0, logits_bytes: 0,
                sampling_workspace_offset: 0, sampling_workspace_bytes: 0,
                sg_detect_offset: 0, sg_knowledge_offset: 0,
                sg_data_bytes: 0,
                total_scratchpad_bytes: 0,
            },
            num_layers: 1,
            vocab_size: 100,
            hidden: 64,
            rope_cache: None,
            total_scratchpad_bytes: 256,
            logits_scratch_offset: 128,
            hetero_layout: None,
            tensor_sources: std::collections::HashMap::new(),
            source_map: None,
        };
        // Verify field access works
        assert_eq!(output.num_layers, 1);
        assert_eq!(output.vocab_size, 100);
        assert_eq!(output.hidden, 64);
        assert_eq!(output.total_scratchpad_bytes, 256);
        assert!(output.layer_code.code_size() > 0);
    }

    #[test]
    fn test_buffer_layout_default_values() {
        let layout = mega_kernel_abi::BufferLayout {
            activation_a_offset: 0, activation_b_offset: 0,
            activation_bytes: 0,
            logits_offset: 0, logits_bytes: 0,
            sampling_workspace_offset: 0, sampling_workspace_bytes: 0,
            sg_detect_offset: 0, sg_knowledge_offset: 0,
            sg_data_bytes: 0,
            total_scratchpad_bytes: 0,
        };
        assert_eq!(layout.activation_bytes, 0);
        assert_eq!(layout.logits_bytes, 0);
        assert_eq!(layout.sg_data_bytes, 0);
        assert_eq!(layout.total_scratchpad_bytes, 0);
    }

    #[test]
    fn test_gemm_microkernel_strategy_variants() {
        use planner::GemmMicrokernelStrategy;
        let strategies = vec![
            GemmMicrokernelStrategy::BlisAvx2,
            GemmMicrokernelStrategy::Scalar,
        ];
        for s in &strategies {
            let _ = format!("{:?}", s);
        }
        assert_ne!(strategies[0], strategies[1]);
    }

    #[test]
    fn test_execution_plan_from_profile() {
        let profile = DeviceProfile::detect();
        let plan = planner::ExecutionPlan::from_profile(&profile);
        // Just verify it constructs without panic
        let _ = format!("{:?}", plan.gemm_blocking);
    }

    #[test]
    fn test_compiler_clear_disk_cache_no_panic() {
        let profile = DeviceProfile::detect();
        let mut compiler = InferenceCompiler::with_profile(profile);
        // Should not panic even when cache is empty
        compiler.clear_disk_cache();
        assert_eq!(compiler.cache_size(), 0);
    }

    #[test]
    fn test_layer_ir_from_model_config_fields() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 4);
        assert_eq!(ir.hidden, config.hidden_size);
        assert_eq!(ir.num_heads, config.num_heads);
        assert_eq!(ir.max_batch, 4);
    }

    #[test]
    fn test_per_layer_weight_layout_all_zeros() {
        let layout = mega_kernel_abi::PerLayerWeightLayout {
            attn_norm_offset: 0, attn_norm_bytes: 0,
            w_q_offset: 0, w_q_bytes: 0,
            w_k_offset: 0, w_k_bytes: 0,
            w_v_offset: 0, w_v_bytes: 0,
            w_o_offset: 0, w_o_bytes: 0,
            w_q_norm_offset: 0, w_q_norm_bytes: 0,
            w_k_norm_offset: 0, w_k_norm_bytes: 0,
            ffn_norm_offset: 0, ffn_norm_bytes: 0,
            w_gate_offset: 0, w_gate_bytes: 0,
            w_up_offset: 0, w_up_bytes: 0,
            w_down_offset: 0, w_down_bytes: 0,
        };
        assert_eq!(layout.w_q_offset, 0);
        assert_eq!(layout.w_down_bytes, 0);
    }

    // ── 10 additional tests (wave-12k76) ─────────────────────────────────

    /// Verify FfnFusionStrategy variants are distinct, Copy-able, and Debug-able.
    #[test]
    fn test_ffn_fusion_strategy_variants() {
        // Arrange
        let inject = planner::FfnFusionStrategy::GateSiLUInject;
        let separate = planner::FfnFusionStrategy::SeparateGemm;

        // Assert: distinct variants
        assert_ne!(inject, separate);

        // Assert: Copy semantics (reuse after move)
        let inject_copy = inject;
        assert_eq!(inject, inject_copy);

        // Assert: Debug formatting
        let debug = format!("{:?}", inject);
        assert!(debug.contains("GateSiLUInject"));
        let debug2 = format!("{:?}", separate);
        assert!(debug2.contains("SeparateGemm"));
    }

    /// Verify AttentionVariant has all expected CPU variants and they are
    /// distinct, Copy-able, and PartialEq.
    #[test]
    fn test_attention_variant_cpu_paths_distinct() {
        // Arrange
        let amx = planner::AttentionVariant::AmxTile;
        let avx512 = planner::AttentionVariant::Avx512Loop;
        let neon = planner::AttentionVariant::NeonLoop;
        let scalar = planner::AttentionVariant::ScalarLoop;

        // Assert: all CPU variants pairwise distinct
        let variants = [amx, avx512, neon, scalar];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j],
                    "AttentionVariant at index {} and {} must differ", i, j);
            }
        }

        // Assert: Copy works
        let amx_copy = amx;
        assert_eq!(amx, amx_copy);

        // Assert: Debug produces variant names
        assert!(format!("{:?}", amx).contains("AmxTile"));
        assert!(format!("{:?}", scalar).contains("ScalarLoop"));
    }

    /// Verify GpuSmPartition field construction and Debug output.
    #[test]
    fn test_gpu_sm_partition_fields_and_debug() {
        // Arrange & Act
        let partition = planner::GpuSmPartition {
            total_sm: 108,
            num_partitions: 3,
            sm_per_partition: 36,
        };

        // Assert: field values preserved
        assert_eq!(partition.total_sm, 108);
        assert_eq!(partition.num_partitions, 3);
        assert_eq!(partition.sm_per_partition, 36);

        // Assert: invariant — total = partitions * sm_per_partition
        assert_eq!(partition.total_sm, partition.num_partitions * partition.sm_per_partition);

        // Assert: Debug formats without panic
        let debug = format!("{:?}", partition);
        assert!(debug.contains("total_sm"));
        assert!(debug.contains("108"));
    }

    /// Verify NumaBinding field construction and Debug output.
    #[test]
    fn test_numa_binding_fields_and_debug() {
        // Arrange & Act
        let binding = planner::NumaBinding {
            node_id: 1,
            core_start: 0,
            core_end: 16,
            l3_bytes: 32 * 1024 * 1024,
        };

        // Assert: field access
        assert_eq!(binding.node_id, 1);
        assert_eq!(binding.core_end - binding.core_start, 16);
        assert_eq!(binding.l3_bytes, 33_554_432);

        // Assert: Debug
        let debug = format!("{:?}", binding);
        assert!(debug.contains("node_id"));
    }

    /// Verify FusionStrategy fields carry meaningful defaults and the struct
    /// can be constructed with struct literal syntax.
    #[test]
    fn test_fusion_strategy_fields_constructible() {
        // Arrange & Act
        let strategy = planner::FusionStrategy {
            max_epilogue_depth: 3,
            tile_fusion_threshold: 2048,
            ffn_strategy: planner::FfnFusionStrategy::GateSiLUInject,
            norm_into_gemm: true,
            qkv_shared_input: true,
            cross_layer_residual: false,
        };

        // Assert: field values
        assert_eq!(strategy.max_epilogue_depth, 3);
        assert_eq!(strategy.tile_fusion_threshold, 2048);
        assert_eq!(strategy.ffn_strategy, planner::FfnFusionStrategy::GateSiLUInject);
        assert!(strategy.norm_into_gemm);
        assert!(strategy.qkv_shared_input);
        assert!(!strategy.cross_layer_residual);

        // Assert: Debug
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("max_epilogue_depth"));
    }

    /// Verify AttentionPlan construction with struct literal and that
    /// all fields are readable.
    #[test]
    fn test_attention_plan_field_access() {
        // Arrange & Act
        let plan = planner::AttentionPlan {
            variant: planner::AttentionVariant::Avx512Loop,
            tile_q: 64,
            tile_kv: 64,
            online_softmax: true,
            warp_specialization: false,
            tma_enabled: false,
        };

        // Assert: field values
        assert_eq!(plan.variant, planner::AttentionVariant::Avx512Loop);
        assert_eq!(plan.tile_q, 64);
        assert_eq!(plan.tile_kv, 64);
        assert!(plan.online_softmax);
        assert!(!plan.warp_specialization);
        assert!(!plan.tma_enabled);

        // Assert: Debug
        let debug = format!("{:?}", plan);
        assert!(debug.contains("Avx512Loop"));
    }

    /// Verify CacheBudgetPlan fields are constructible and arithmetic
    /// on budgets does not overflow.
    #[test]
    fn test_cache_budget_plan_sum_no_overflow() {
        // Arrange
        let plan = planner::CacheBudgetPlan {
            l1_tile_budget: 16 * 1024,
            l1_fusion_scratch: 4 * 1024,
            l2_kv_budget: 512 * 1024,
            l2_weight_budget: 1024 * 1024,
            l2_activation_budget: 256 * 1024,
            l3_model_budget: 8 * 1024 * 1024 * 1024,
            l3_kv_cold_budget: 2 * 1024 * 1024 * 1024,
        };

        // Act: sum L2 budgets
        let l2_total = plan.l2_kv_budget
            .checked_add(plan.l2_weight_budget)
            .and_then(|s| s.checked_add(plan.l2_activation_budget))
            .expect("L2 budget sum overflow");

        // Assert: sums are reasonable
        assert!(l2_total > 0);
        assert!(plan.l1_tile_budget > 0);
        assert!(plan.l3_model_budget > plan.l3_kv_cold_budget);

        // Assert: Debug
        let _ = format!("{:?}", plan);
    }

    /// Verify MegaKernelParams constant length matches
    /// MEGA_KERNEL_STACK_OFFSETS length + 6 (register params).
    #[test]
    fn test_mega_kernel_params_and_offsets_consistency() {
        // Arrange: constants from mega_kernel_abi
        let param_count = mega_kernel_abi::MEGA_KERNEL_PARAMS.len();
        let stack_offset_count = mega_kernel_abi::MEGA_KERNEL_STACK_OFFSETS.len();

        // Assert: total params = 6 register params + stack params
        assert_eq!(param_count, 6 + stack_offset_count,
            "MEGA_KERNEL_PARAMS ({}) must equal 6 + MEGA_KERNEL_STACK_OFFSETS ({})",
            param_count, stack_offset_count);

        // Assert: well-known first and last param names
        assert_eq!(mega_kernel_abi::MEGA_KERNEL_PARAMS[0], "input_ids_ptr");
        assert_eq!(mega_kernel_abi::MEGA_KERNEL_PARAMS[5], "batch_size");
        assert_eq!(mega_kernel_abi::MEGA_KERNEL_PARAMS[param_count - 1], "batch_ctx_ptr");

        // Assert: stack offsets are strictly increasing by 8
        for w in mega_kernel_abi::MEGA_KERNEL_STACK_OFFSETS.windows(2) {
            assert_eq!(w[1] - w[0], 8,
                "stack offsets must increase by 8: {} -> {}", w[0], w[1]);
        }

        // Assert: first stack offset is 16 (after rbp return address)
        assert_eq!(mega_kernel_abi::MEGA_KERNEL_STACK_OFFSETS[0], 16);
    }

    /// Verify OutputMode variants are constructible and Debug-able.
    #[test]
    fn test_output_mode_variants_construction() {
        // Arrange & Act
        let generate = mega_kernel_abi::OutputMode::Generate {
            max_new_tokens: 512,
            eos_token_id: 2,
        };
        let classify_binary = mega_kernel_abi::OutputMode::ClassifyBinary {
            positive_token_id: 1,
            negative_token_id: 0,
        };
        let classify_multiway = mega_kernel_abi::OutputMode::ClassifyMultiway {
            label_token_ids: vec![100, 200, 300],
        };
        let encode = mega_kernel_abi::OutputMode::EncodeToLayer {
            anchor_layer: 16,
            pool_mode: mega_kernel_abi::PoolMode::MeanPool,
        };

        // Assert: Debug formatting
        let gen_debug = format!("{:?}", generate);
        assert!(gen_debug.contains("Generate"));
        assert!(gen_debug.contains("512"));

        let bin_debug = format!("{:?}", classify_binary);
        assert!(bin_debug.contains("ClassifyBinary"));

        let multi_debug = format!("{:?}", classify_multiway);
        assert!(multi_debug.contains("ClassifyMultiway"));

        let enc_debug = format!("{:?}", encode);
        assert!(enc_debug.contains("EncodeToLayer"));
        assert!(enc_debug.contains("MeanPool"));

        // Assert: PoolMode variants
        let pool_modes = [
            format!("{:?}", mega_kernel_abi::PoolMode::LastToken),
            format!("{:?}", mega_kernel_abi::PoolMode::MeanPool),
            format!("{:?}", mega_kernel_abi::PoolMode::ClsToken),
        ];
        for name in &pool_modes {
            assert!(!name.is_empty(), "PoolMode variant debug must not be empty");
        }
    }

    /// Verify compute_hash differs for different batch sizes.
    #[test]
    fn test_compute_hash_differs_for_different_batch() {
        // Arrange
        let config = ModelConfig::llama_7b();
        let profile = DeviceProfile::detect();
        let compiler = InferenceCompiler::with_profile(profile);

        let ir_batch1 = LayerIR::from_model_config(&config, 1);
        let ir_batch4 = LayerIR::from_model_config(&config, 4);

        // Act
        let h1 = compiler.compute_hash(&ir_batch1);
        let h4 = compiler.compute_hash(&ir_batch4);

        // Assert: different max_batch produces different hash
        assert_ne!(h1, h4, "different max_batch should produce different hashes");
        assert_ne!(h1, 0);
        assert_ne!(h4, 0);
    }

    // ── 10 additional tests (wave-12ka2) ──────────────────────────────────

    /// Verify RooflineResult construction and that all BottleneckClass fields
    /// are individually assignable.
    #[test]
    fn test_roofline_result_fields_and_bottleneck_classes() {
        // Arrange & Act
        let result = planner::RooflineResult {
            ridge_point: 12.5,
            peak_gflops: 2048.0,
            peak_bandwidth_gbs: 200.0,
            gemm_prefill: planner::BottleneckClass::ComputeBound,
            gemm_decode: planner::BottleneckClass::MemoryBound,
            attn_prefill: planner::BottleneckClass::Mixed,
            attn_decode: planner::BottleneckClass::MemoryBound,
            elementwise: planner::BottleneckClass::MemoryBound,
        };

        // Assert: field values preserved
        assert!((result.ridge_point - 12.5).abs() < f64::EPSILON);
        assert_eq!(result.gemm_prefill, planner::BottleneckClass::ComputeBound);
        assert_eq!(result.gemm_decode, planner::BottleneckClass::MemoryBound);
        assert_eq!(result.attn_prefill, planner::BottleneckClass::Mixed);

        // Assert: Debug formats without panic
        let debug = format!("{:?}", result);
        assert!(debug.contains("ridge_point"));
        assert!(debug.contains("ComputeBound"));
    }

    /// Verify FeatureDecision construction, Debug, and Clone.
    #[test]
    fn test_feature_decision_construction_and_clone() {
        // Arrange & Act
        let decision = planner::FeatureDecision {
            feature: "avx512_bf16".into(),
            enabled: true,
            reason: "hardware supports VDPBF16PS".into(),
        };

        // Assert: field access
        assert_eq!(decision.feature, "avx512_bf16");
        assert!(decision.enabled);
        assert!(decision.reason.contains("VDPBF16PS"));

        // Assert: Clone produces equal value
        let cloned = decision.clone();
        assert_eq!(cloned.feature, decision.feature);
        assert_eq!(cloned.enabled, decision.enabled);

        // Assert: Debug
        let debug = format!("{:?}", decision);
        assert!(debug.contains("avx512_bf16"));
    }

    /// Verify FeaturePlan construction, empty decisions vec, and budget tracking.
    #[test]
    fn test_feature_plan_empty_and_with_decisions() {
        // Arrange: empty plan
        let empty = planner::FeaturePlan {
            decisions: Vec::new(),
            l1i_used: 0,
            l1i_budget: 32 * 1024,
        };

        // Assert: empty state
        assert!(empty.decisions.is_empty());
        assert_eq!(empty.l1i_used, 0);
        assert!(empty.l1i_budget > 0);

        // Act: plan with decisions
        let with_features = planner::FeaturePlan {
            decisions: vec![
                planner::FeatureDecision {
                    feature: "tma_2d".into(),
                    enabled: false,
                    reason: "SM < 90".into(),
                },
            ],
            l1i_used: 1024,
            l1i_budget: 32 * 1024,
        };

        // Assert: populated state
        assert_eq!(with_features.decisions.len(), 1);
        assert!(!with_features.decisions[0].enabled);
        assert!(with_features.l1i_used < with_features.l1i_budget);
    }

    /// Verify BatchPlan field construction and arithmetic on budget fields.
    #[test]
    fn test_batch_plan_fields_and_arithmetic() {
        // Arrange & Act
        let plan = planner::BatchPlan {
            decode_ratio_cap: 0.7,
            max_chunk_size: 4096,
            golden_sizes: vec![128, 256, 512, 1024],
            min_compact_threshold: 4,
            compact_waste_threshold: 0.3,
            decode_slots: 64,
            max_chunks_per_batch: 8,
        };

        // Assert: field values
        assert!((plan.decode_ratio_cap - 0.7).abs() < f32::EPSILON);
        assert_eq!(plan.max_chunk_size, 4096);
        assert_eq!(plan.golden_sizes.len(), 4);
        assert_eq!(plan.decode_slots, 64);

        // Assert: golden_sizes is sorted ascending
        for w in plan.golden_sizes.windows(2) {
            assert!(w[0] < w[1], "golden_sizes must be sorted ascending");
        }

        // Assert: Debug
        let _ = format!("{:?}", plan);
    }

    /// Verify ParallelPlan construction with CPU NUMA bindings and no GPU partition.
    #[test]
    fn test_parallel_plan_cpu_numa_bindings() {
        // Arrange & Act
        let plan = planner::ParallelPlan {
            wave_count: 2,
            gpu_sm_partition: None,
            numa_bindings: vec![
                planner::NumaBinding { node_id: 0, core_start: 0, core_end: 16, l3_bytes: 32 * 1024 * 1024 },
                planner::NumaBinding { node_id: 1, core_start: 16, core_end: 32, l3_bytes: 32 * 1024 * 1024 },
            ],
            min_batch_tokens_per_wave: 256,
            min_decode_per_wave: 8,
            occupancy_target: 0.85,
        };

        // Assert: CPU path — no GPU partition
        assert!(plan.gpu_sm_partition.is_none());
        assert_eq!(plan.numa_bindings.len(), 2);
        assert_eq!(plan.wave_count, 2);

        // Assert: core ranges are non-overlapping
        assert!(plan.numa_bindings[0].core_end <= plan.numa_bindings[1].core_start);

        // Assert: occupancy in valid range
        assert!(plan.occupancy_target > 0.0 && plan.occupancy_target <= 1.0);
    }

    /// Verify BusinessConfig default has sensible values and
    /// all boolean flags default to false.
    #[test]
    fn test_mega_kernel_business_config_defaults() {
        // Act
        let config = mega_kernel_abi::BusinessConfig::default();

        // Assert: output_modes has default Generate
        assert_eq!(config.output_modes.len(), 1);
        assert!(matches!(&config.output_modes[0],
            mega_kernel_abi::OutputMode::Generate { max_new_tokens: 512, eos_token_id: 2 }));

        // Assert: all boolean flags default to false
        assert!(!config.guardrail_enabled);
        assert!(config.semantic_gatekeeper.is_none());
        assert!(config.intent_anchor_layer.is_none());
        assert!(config.cot_step_hook.is_none());
        assert!(!config.session_enabled);
        assert!(!config.multimodal_enabled);
        assert!(!config.debug_jit);
    }

    /// Verify CotStepConfig construction and field access.
    #[test]
    fn test_cot_step_config_construction() {
        // Arrange & Act
        let config = mega_kernel_abi::CotStepConfig {
            shared_mem_offset: 0x1000,
        };

        // Assert: field value preserved
        assert_eq!(config.shared_mem_offset, 0x1000);

        // Assert: Debug formats without panic
        let debug = format!("{:?}", config);
        assert!(debug.contains("shared_mem_offset"));
    }

    /// Verify MtpKernelConfig construction, field arithmetic, and Debug.
    #[test]
    fn test_mtp_kernel_config_construction_and_arithmetic() {
        // Arrange & Act
        let config = mega_kernel_abi::MtpKernelConfig {
            depth: 4,
            hidden_size: 4096,
            vocab_size: 32000,
        };

        // Assert: field values
        assert_eq!(config.depth, 4);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.vocab_size, 32000);

        // Act: compute weight bytes per depth (F32)
        let weight_bytes_per_depth = config.hidden_size * config.vocab_size * 4;

        // Assert: arithmetic does not overflow for realistic values
        assert!(weight_bytes_per_depth > 0);
        assert_eq!(weight_bytes_per_depth, 4096 * 32000 * 4);

        // Assert: total MTP weight = depth * per-depth bytes
        let total = config.depth * weight_bytes_per_depth;
        assert!(total > weight_bytes_per_depth);

        // Assert: Debug
        let debug = format!("{:?}", config);
        assert!(debug.contains("depth"));
        assert!(debug.contains("4096"));
    }

    // ── 10 additional tests (wave-12x88) ──────────────────────────────────

    /// Verify CompilerGraph produces a substantial tensor count for LLaMA-7B.
    #[test]
    fn test_compiler_graph_tensor_count_positive() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        assert!(
            graph.tensors.len() > 10,
            "LLaMA-7B should have significant tensors, got {}",
            graph.tensors.len(),
        );
    }

    /// Verify topological sort returns exactly num_ops unique entries.
    #[test]
    fn test_compiler_graph_topo_sort_completeness() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        let topo = graph.topological_sort();
        assert_eq!(topo.len(), graph.num_ops(), "topo length must match num_ops");

        let unique: std::collections::HashSet<_> = topo.iter().collect();
        assert_eq!(unique.len(), topo.len(), "topo sort must have no duplicates");
    }

    /// Verify LayerIR head_dim is derived as hidden / num_heads.
    #[test]
    fn test_layer_ir_head_dim_derived_from_hidden_and_heads() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);

        assert_eq!(
            ir.head_dim,
            ir.hidden / ir.num_heads,
            "head_dim must equal hidden / num_heads",
        );
        assert_eq!(ir.head_dim, 128, "LLaMA-7B head_dim should be 128");
    }

    /// Verify fusion plan produces fewer groups than raw ops.
    #[test]
    fn test_fusion_plan_llama7b_reduces_group_count() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        let registry = ScalarOpRegistry::with_defaults();
        let fplan = fusion::fuse_with_dag(&graph, &registry, &exec_plan);

        assert!(
            fplan.num_groups() < graph.num_ops(),
            "fusion should reduce {} ops to fewer groups (got {})",
            graph.num_ops(), fplan.num_groups(),
        );
    }

    /// Verify ExecutionPlan::build succeeds for LLaMA-7B with default bias.
    #[test]
    fn test_execution_plan_build_with_default_bias() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();

        let plan = ExecutionPlan::build(&ir, &profile, &planner::StrategyBias::default());
        // Verify gemm_blocking is accessible and Debug-able
        let _ = format!("{:?}", plan.gemm_blocking);
    }

    /// Verify compute_hash is deterministic across different compiler instances.
    #[test]
    fn test_compute_hash_independent_of_compiler_instance() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();

        let compiler1 = InferenceCompiler::with_profile(profile.clone());
        let compiler2 = InferenceCompiler::with_profile(profile);

        let h1 = compiler1.compute_hash(&ir);
        let h2 = compiler2.compute_hash(&ir);
        assert_eq!(
            h1, h2,
            "same profile and IR must produce same hash regardless of compiler instance",
        );
    }

    /// Verify CompilerGraph for LLaMA-7B contains Gemm ops.
    #[test]
    fn test_compiler_graph_contains_gemm_ops() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        let topo = graph.topological_sort();
        let mut gemm_count = 0;
        for &op_id in &topo {
            if let Some(op) = graph.op(op_id) {
                if matches!(op.op_resolved(&graph), Some(Op::Gemm(_))) {
                    gemm_count += 1;
                }
            }
        }
        assert!(gemm_count > 0, "LLaMA-7B should have Gemm ops, found {}", gemm_count);
    }

    /// Verify graph.infer_computation_dtype() returns a dtype with positive size.
    #[test]
    fn test_graph_infer_dtype_size_positive() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        let dtype = graph.infer_computation_dtype();
        assert!(dtype.size_bytes() > 0, "computation dtype must have positive size");
    }

    /// Verify graph_content_hash is identical for a cloned CompilerGraph.
    #[test]
    fn test_graph_content_hash_same_for_cloned_graph() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let graph_clone = graph.clone();

        let compiler = InferenceCompiler::with_profile(profile);
        let h1 = compiler.graph_content_hash(&graph);
        let h2 = compiler.graph_content_hash(&graph_clone);
        assert_eq!(h1, h2, "cloned graph must produce identical content hash");
    }

    // ── 10 additional tests (wave-13x01) ──────────────────────────────────

    /// Verify CompilerGraph::new() produces a genuinely empty graph.
    #[test]
    fn test_empty_graph_has_zero_ops_and_tensors() {
        // Act
        let graph = CompilerGraph::new();

        // Assert
        assert_eq!(graph.num_ops(), 0, "empty graph must have zero ops");
        assert_eq!(graph.num_tensors(), 0, "empty graph must have zero tensors");
        assert!(graph.inputs.is_empty(), "empty graph must have no inputs");
        assert!(graph.outputs.is_empty(), "empty graph must have no outputs");
        assert_eq!(graph.max_seq_len, 2048, "default max_seq_len should be 2048");
    }

    /// Verify Default trait produces the same result as ::new().
    #[test]
    fn test_graph_default_matches_new() {
        // Act
        let from_new = CompilerGraph::new();
        let from_default = CompilerGraph::default();

        // Assert: structural equality
        assert_eq!(from_new.num_ops(), from_default.num_ops());
        assert_eq!(from_new.num_tensors(), from_default.num_tensors());
        assert_eq!(from_new.max_seq_len, from_default.max_seq_len);
        assert_eq!(from_new.inputs.len(), from_default.inputs.len());
        assert_eq!(from_new.outputs.len(), from_default.outputs.len());
    }

    /// Verify single-node graph: add one tensor + one op, check def-use chains.
    #[test]
    fn test_single_node_graph_def_use_chains() {
        // Arrange
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor_concrete("input", &[4, 4], DType::F32);
        let t_out = graph.add_tensor_concrete("output", &[4, 4], DType::F32);
        graph.inputs.push(t_in);

        // Act
        let op_id = graph.add_op(Op::Silu, vec![t_in], vec![t_out], "silu");

        // Assert: producer set on output
        let out_meta = graph.tensor(t_out).expect("output tensor must exist");
        assert_eq!(out_meta.producer, Some(op_id), "output producer must be the Silu op");

        // Assert: consumer set on input
        let in_meta = graph.tensor(t_in).expect("input tensor must exist");
        assert_eq!(in_meta.consumers.len(), 1, "input must have exactly one consumer");
        assert_eq!(in_meta.consumers[0], op_id);
    }

    /// Verify topological sort on a 4-op chain produces correct dependency order.
    #[test]
    fn test_four_node_chain_topological_order() {
        // Arrange: norm → gemm → silu → residual
        let mut graph = CompilerGraph::new();
        let t0 = graph.add_tensor_concrete("act", &[1, 128], DType::F32);
        let t1 = graph.add_tensor_concrete("normed", &[1, 128], DType::F32);
        let t2 = graph.add_tensor_concrete("weight", &[128, 512], DType::F32);
        let t3 = graph.add_tensor_concrete("proj", &[1, 512], DType::F32);
        let t4 = graph.add_tensor_concrete("activated", &[1, 512], DType::F32);
        let t5 = graph.add_tensor_concrete("res_out", &[1, 512], DType::F32);
        let t_skip = graph.add_tensor_concrete("skip", &[1, 512], DType::F32);
        graph.inputs.extend_from_slice(&[t0, t2, t_skip]);

        let op_norm = graph.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }), vec![t0], vec![t1], "norm",
        );
        let op_gemm = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 512, k: 128, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![t1, t2], vec![t3], "gemm",
        );
        let op_silu = graph.add_op(Op::Silu, vec![t3], vec![t4], "silu");
        let op_res = graph.add_op(Op::Residual, vec![t4, t_skip], vec![t5], "residual");

        // Act
        let topo = graph.topological_sort();

        // Assert: all 4 ops present
        assert_eq!(topo.len(), 4, "must have exactly 4 ops in topo sort");

        // Assert: dependency order — each op appears after its dependencies
        let pos = |id: crate::compiler::graph::OpId| {
            topo.iter().position(|&x| x == id).expect("op must be in topo sort")
        };
        assert!(pos(op_norm) < pos(op_gemm), "norm before gemm");
        assert!(pos(op_gemm) < pos(op_silu), "gemm before silu");
        assert!(pos(op_silu) < pos(op_res), "silu before residual");
    }

    /// Verify tensor_numel for concrete vs symbolic shapes.
    #[test]
    fn test_tensor_numel_concrete_vs_symbolic() {
        // Arrange
        let mut graph = CompilerGraph::new();
        let t_concrete = graph.add_tensor_concrete("c", &[2, 4, 8], DType::F32);
        let t_symbolic = graph.add_tensor(
            "s",
            vec![SymDim::Symbolic { name: "seq".into(), max_value: Some(512) }, SymDim::Concrete(64)],
            DType::F32,
        );

        // Act & Assert: concrete numel = 2*4*8 = 64
        assert_eq!(graph.tensor_numel(t_concrete), Some(64));

        // Assert: symbolic numel treats Symbolic as 0 → product = 0 → max(1) = 1
        assert_eq!(graph.tensor_numel(t_symbolic), Some(1));

        // Assert: numel_for_alloc uses max_value 512 → 512*64 = 32768
        assert_eq!(graph.tensor_numel_for_alloc(t_symbolic, 2048), Some(32768));
    }

    /// Verify Display trait formats graph with op count and tensor count.
    #[test]
    fn test_graph_display_format() {
        // Arrange
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        // Act
        let display = format!("{}", graph);

        // Assert: must contain header with op and tensor counts
        assert!(display.contains("CompilerGraph:"), "Display must contain header");
        assert!(display.contains("ops"), "Display must mention ops");
        assert!(display.contains("tensors"), "Display must mention tensors");

        // Assert: each op line contains the op label
        let num_lines = display.lines().count();
        assert!(num_lines >= 2, "Display must have header + at least one op line");
    }

    /// Verify Debug trait on CompilerGraph produces output without panic.
    #[test]
    fn test_graph_debug_no_panic() {
        // Arrange
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        // Act
        let debug = format!("{:?}", graph);

        // Assert: Debug output is non-empty and contains key field names
        assert!(!debug.is_empty(), "Debug output must not be empty");
        assert!(debug.contains("ops"), "Debug must contain 'ops' field");
        assert!(debug.contains("tensors"), "Debug must contain 'tensors' field");
    }

    /// Verify weight_layout returns offsets for all non-activation inputs.
    #[test]
    fn test_weight_layout_covers_all_weight_inputs() {
        // Arrange
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

        // Act
        let layout = graph.weight_layout();

        // Assert: number of offsets = inputs.len() - 1 (skip activation input)
        let expected_count = graph.inputs.len().saturating_sub(1);
        assert_eq!(
            layout.offsets.len(), expected_count,
            "weight_layout must cover all non-activation inputs",
        );

        // Assert: total_bytes is positive
        assert!(layout.total_bytes > 0, "total weight bytes must be positive");

        // Assert: offsets are strictly increasing (contiguous layout)
        for window in layout.offsets.windows(2) {
            assert!(window[0].1 < window[1].1, "weight offsets must be strictly increasing");
        }
    }

    /// Verify compute_hash is sensitive to hidden dimension changes.
    #[test]
    fn test_compute_hash_differs_when_hidden_changes() {
        // Arrange
        let config1 = ModelConfig::llama_7b();
        let mut config2 = ModelConfig::llama_7b();
        config2.hidden_size = 8192; // different hidden

        let ir1 = LayerIR::from_model_config(&config1, 1);
        let ir2 = LayerIR::from_model_config(&config2, 1);
        let profile = DeviceProfile::detect();
        let compiler = InferenceCompiler::with_profile(profile);

        // Act
        let h1 = compiler.compute_hash(&ir1);
        let h2 = compiler.compute_hash(&ir2);

        // Assert
        assert_ne!(h1, h2, "different hidden_size must produce different hashes");
    }

    /// Verify topological sort on empty graph returns empty vec.
    #[test]
    fn test_topological_sort_empty_graph() {
        // Arrange
        let graph = CompilerGraph::new();

        // Act
        let topo = graph.topological_sort();

        // Assert
        assert!(topo.is_empty(), "empty graph topo sort must return empty vec");
    }

    // ── 10 additional tests (wave-13x02) ──────────────────────────────────

    /// Verify CompilerGraph::add_tensor returns distinct TensorIds.
    #[test]
    fn test_add_tensor_returns_distinct_ids() {
        // Arrange
        let mut graph = CompilerGraph::new();

        // Act
        let t0 = graph.add_tensor_concrete("a", &[4, 4], DType::F32);
        let t1 = graph.add_tensor_concrete("b", &[4, 4], DType::F32);
        let t2 = graph.add_tensor_concrete("c", &[4, 4], DType::F32);

        // Assert: all IDs are unique
        assert_ne!(t0, t1, "tensor IDs must be unique");
        assert_ne!(t1, t2, "tensor IDs must be unique");
        assert_ne!(t0, t2, "tensor IDs must be unique");

        // Assert: tensor count matches
        assert_eq!(graph.num_tensors(), 3);
    }

    /// Verify CompilerGraph::add_op returns distinct OpIds.
    #[test]
    fn test_add_op_returns_distinct_ids() {
        // Arrange
        let mut graph = CompilerGraph::new();
        let t0 = graph.add_tensor_concrete("a", &[4, 4], DType::F32);
        let t1 = graph.add_tensor_concrete("b", &[4, 4], DType::F32);
        let t2 = graph.add_tensor_concrete("c", &[4, 4], DType::F32);

        // Act
        let op0 = graph.add_op(Op::Silu, vec![t0], vec![t1], "silu0");
        let op1 = graph.add_op(Op::Silu, vec![t1], vec![t2], "silu1");

        // Assert: all op IDs are unique
        assert_ne!(op0, op1, "op IDs must be unique");
        assert_eq!(graph.num_ops(), 2);
    }

    /// Verify CompilerGraph::op() returns None for invalid OpId.
    #[test]
    fn test_op_returns_none_for_invalid_id() {
        // Arrange
        let graph = CompilerGraph::new();
        let invalid_id = crate::compiler::graph::OpId(999);

        // Act
        let result = graph.op(invalid_id);

        // Assert
        assert!(result.is_none(), "invalid OpId must return None");
    }

    /// Verify CompilerGraph::tensor() returns None for invalid TensorId.
    #[test]
    fn test_tensor_returns_none_for_invalid_id() {
        // Arrange
        let graph = CompilerGraph::new();
        let invalid_id = TensorId(999);

        // Act
        let result = graph.tensor(invalid_id);

        // Assert
        assert!(result.is_none(), "invalid TensorId must return None");
    }

    /// Verify LayerIR::from_model_config sets correct dtype from config.
    #[test]
    fn test_layer_ir_dtype_matches_config() {
        // Arrange
        let config = ModelConfig::llama_7b();

        // Act
        let ir = LayerIR::from_model_config(&config, 1);

        // Assert: dtype should match config's dtype (BF16 for LLaMA-7B)
        assert_eq!(ir.dtype, config.dtype, "LayerIR dtype must match ModelConfig dtype");
    }

    /// Verify LayerIR::from_model_config sets correct intermediate size.
    #[test]
    fn test_layer_ir_intermediate_matches_config() {
        // Arrange
        let config = ModelConfig::llama_7b();

        // Act
        let ir = LayerIR::from_model_config(&config, 1);

        // Assert: intermediate should match config's intermediate_size
        assert_eq!(
            ir.intermediate, config.intermediate_size,
            "LayerIR intermediate must match ModelConfig intermediate_size",
        );
    }

    /// Verify InferenceCompiler::with_profile uses provided profile.
    #[test]
    fn test_with_profile_uses_provided_profile() {
        // Arrange
        let profile = DeviceProfile::detect();

        // Act
        let compiler = InferenceCompiler::with_profile(profile.clone());

        // Assert: cache starts empty
        assert_eq!(compiler.cache_size(), 0);

        // Assert: can compile with this profile (REQ-UMK-001: unified compile() entry point)
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let mk_config = mega_kernel_abi::CompileConfig {
            max_seq_len: 512,
            debug_jit: false,
            hetero: None,
            target: mega_kernel_abi::CompileTarget::Cpu,
        };
        let mut compiler_mut = InferenceCompiler::with_profile(profile);
        let result = compiler_mut.compile(graph, &mk_config, None);
        assert!(result.is_ok(), "compile should succeed with provided profile");
    }

    /// Verify CompilerGraph with single Gemm op has correct topology.
    #[test]
    fn test_single_gemm_graph_topo_sort() {
        // Arrange
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor_concrete("input", &[1, 128], DType::F32);
        let t_w = graph.add_tensor_concrete("weight", &[128, 256], DType::F32);
        let t_out = graph.add_tensor_concrete("output", &[1, 256], DType::F32);
        graph.inputs.extend_from_slice(&[t_in, t_w]);

        // Act
        let _op_id = graph.add_op(Op::Gemm(GemmSpec { m: SymDim::Concrete(1), n: 256, k: 128, dtype: DType::F32, trans_b: false, has_bias: false }),
            vec![t_in, t_w],
            vec![t_out],
            "gemm",
        );

        // Assert: single op in topo sort
        let topo = graph.topological_sort();
        assert_eq!(topo.len(), 1, "single Gemm graph should have exactly 1 op in topo sort");

        // Assert: output tensor has producer set
        let out_meta = graph.tensor(t_out).expect("output must exist");
        assert!(out_meta.producer.is_some(), "output must have a producer");
    }

    /// Verify CompilerError variants are Debug-able and Display-able.
    #[test]
    fn test_compiler_error_variants_debug_and_display() {
        // Arrange
        let errors: Vec<crate::types::CompilerError> = vec![
            crate::types::CompilerError::InvalidGraph("cycle detected".into()),
            crate::types::CompilerError::CodegenViolation("alignment".into()),
            crate::types::CompilerError::FeatureDisabled("jit-x86".into()),
        ];

        // Act & Assert: Debug and Display do not panic, and outputs are distinct
        let mut debug_outputs = Vec::new();
        let mut display_outputs = Vec::new();
        for err in &errors {
            let debug = format!("{:?}", err);
            let display = format!("{}", err);
            assert!(!debug.is_empty(), "Debug output must not be empty");
            assert!(!display.is_empty(), "Display output must not be empty");
            debug_outputs.push(debug);
            display_outputs.push(display);
        }

        // Assert: each variant produces distinct debug output
        assert_ne!(debug_outputs[0], debug_outputs[1], "different variants must produce different Debug");
        assert_ne!(debug_outputs[1], debug_outputs[2], "different variants must produce different Debug");

        // Assert: Display contains key substrings
        assert!(display_outputs[0].contains("invalid graph"), "InvalidGraph Display must contain 'invalid graph'");
        assert!(display_outputs[1].contains("codegen violation"), "CodegenViolation Display must contain 'codegen violation'");
        assert!(display_outputs[2].contains("feature disabled"), "FeatureDisabled Display must contain 'feature disabled'");
    }

    // ── 10 additional tests (wave-13x03) ──────────────────────────────────

    /// Verify CacheStats construction with explicit values and that all
    /// fields are individually accessible.
    #[test]
    fn test_cache_stats_field_access() {
        // Arrange & Act
        let stats = cache::CacheStats {
            memory_hits: 42,
            disk_hits: 7,
            misses: 3,
            total_code_bytes: 1024 * 512,
            num_entries: 10,
        };

        // Assert: each field preserves its value
        assert_eq!(stats.memory_hits, 42);
        assert_eq!(stats.disk_hits, 7);
        assert_eq!(stats.misses, 3);
        assert!(stats.total_code_bytes > 0);
        assert_eq!(stats.num_entries, 10);

        // Assert: total operations = hits + misses
        let total_ops = stats.memory_hits + stats.disk_hits + stats.misses;
        assert_eq!(total_ops, 52);
    }

    /// Verify that CompilationCache::new() reports zero entries and
    /// stats returns all-zero fields.
    #[test]
    fn test_compilation_cache_default_is_empty() {
        // Act
        let cache = CompilationCache::new();

        // Assert: empty
        assert_eq!(cache.len(), 0, "default cache must be empty");

        // Assert: stats reflect empty state
        let stats = cache.stats();
        assert_eq!(stats.num_entries, 0);
        assert_eq!(stats.total_code_bytes, 0);
        assert_eq!(stats.memory_hits, 0);
    }

    /// Verify CompilerGraph::set_quant_weight_bytes affects the weight_layout
    /// calculation — the overridden byte count is used instead of numel*dtype.
    #[test]
    fn test_set_quant_weight_bytes_affects_layout() {
        // Arrange: two weight inputs
        let mut graph = CompilerGraph::new();
        let t_act = graph.add_tensor_concrete("activation", &[1, 64], DType::F32);
        let t_w0 = graph.add_tensor_concrete("weight0", &[64, 128], DType::F32);
        let t_w1 = graph.add_tensor_concrete("weight1", &[128, 256], DType::F32);
        graph.inputs.extend_from_slice(&[t_act, t_w0, t_w1]);

        // Act: override quant weight bytes for weight0
        graph.set_quant_weight_bytes(t_w0, 1024);

        // Assert: weight_layout uses the overridden size for weight0
        let layout = graph.weight_layout();
        let offset_w0 = layout.offset_of(t_w0).expect("weight0 must have offset");
        let offset_w1 = layout.offset_of(t_w1).expect("weight1 must have offset");

        // weight1 offset = weight0 offset + quant override (1024), not numel*4
        assert!(
            offset_w1 >= offset_w0 + 1024,
            "weight1 offset must account for weight0 quant override of 1024",
        );
    }

    /// Verify SymDim::is_symbolic returns correct results for each variant.
    #[test]
    fn test_sym_dim_is_symbolic() {
        // Arrange
        let concrete = SymDim::Concrete(128);
        let symbolic = SymDim::Symbolic { name: "seq".into(), max_value: Some(2048) };

        // Assert
        assert!(!concrete.is_symbolic(), "Concrete should not be symbolic");
        assert!(symbolic.is_symbolic(), "Symbolic should be symbolic");
    }

    /// Verify SymDim::as_concrete returns Some for Concrete and None for Symbolic.
    #[test]
    fn test_sym_dim_as_concrete() {
        // Arrange
        let concrete = SymDim::Concrete(64);
        let symbolic = SymDim::Symbolic { name: "batch".into(), max_value: Some(32) };

        // Assert
        assert_eq!(concrete.as_concrete(), Some(64));
        assert_eq!(symbolic.as_concrete(), None);
    }

    /// Verify ShapeBinding::new is empty, bind adds entries, and get retrieves them.
    #[test]
    fn test_shape_binding_bind_and_get() {
        // Arrange
        let binding = ShapeBinding::new();

        // Assert: empty
        assert!(binding.get("seq_len").is_none(), "empty binding should return None");

        // Act: bind values
        let binding = binding.bind("seq_len", 512).bind("batch_size", 4);

        // Assert: values retrievable
        assert_eq!(binding.get("seq_len"), Some(&512));
        assert_eq!(binding.get("batch_size"), Some(&4));
        assert!(binding.get("nonexistent").is_none());
    }

    /// Verify WeightLayout::offset_of returns correct offset for known tensors
    /// and None for unknown ones.
    #[test]
    fn test_weight_layout_offset_of() {
        // Arrange
        let t0 = TensorId(0);
        let t1 = TensorId(1);
        let t_unknown = TensorId(99);
        let layout = WeightLayout {
            offsets: vec![(t0, 0), (t1, 4096)],
            total_bytes: 8192,
        };

        // Act & Assert: known tensors
        assert_eq!(layout.offset_of(t0), Some(0));
        assert_eq!(layout.offset_of(t1), Some(4096));

        // Assert: unknown tensor
        assert_eq!(layout.offset_of(t_unknown), None);

        // Assert: total_bytes is positive
        assert!(layout.total_bytes > 0);
    }

    /// Verify CompilerGraph::def_use_chains returns correct producer/consumer
    /// relationships for a simple two-op chain.
    #[test]
    fn test_def_use_chains_two_op_chain() {
        // Arrange: norm → silu
        let mut graph = CompilerGraph::new();
        let t_in = graph.add_tensor_concrete("input", &[4, 64], DType::F32);
        let t_mid = graph.add_tensor_concrete("normed", &[4, 64], DType::F32);
        let t_out = graph.add_tensor_concrete("output", &[4, 64], DType::F32);
        graph.inputs.push(t_in);

        let op_norm = graph.add_op(Op::RmsNorm(NormSpec { feature_dim: 4096, eps: 1e-5, dtype: DType::F32, has_weight: true }), vec![t_in], vec![t_mid], "norm");
        let _op_silu = graph.add_op(Op::Silu, vec![t_mid], vec![t_out], "silu");

        // Act
        let chains = graph.def_use_chains();

        // Assert: input has no producer, norm as consumer
        let (prod_in, cons_in) = chains.get(&t_in).expect("input must be in chains");
        assert!(prod_in.is_none(), "graph input has no producer");
        assert_eq!(cons_in.len(), 1, "input consumed by one op");
        assert_eq!(cons_in[0], op_norm);

        // Assert: mid tensor has norm as producer, silu as consumer
        let (prod_mid, cons_mid) = chains.get(&t_mid).expect("mid must be in chains");
        assert_eq!(*prod_mid, Some(op_norm));
        assert_eq!(cons_mid.len(), 1, "mid consumed by one op");
    }

    /// Verify SgConfig construction with all fields, including optional q_tap.
    #[test]
    fn test_sg_config_construction_with_and_without_qtap() {
        // Arrange & Act: without q_tap
        let sg_no_tap = mega_kernel_abi::SgConfig {
            detect_layer: 8,
            inject_offset: 0x2000,
            detect_offset: 0x3000,
            q_tap: None,
        };

        // Assert: field access
        assert_eq!(sg_no_tap.detect_layer, 8);
        assert_eq!(sg_no_tap.inject_offset, 0x2000);
        assert!(sg_no_tap.q_tap.is_none());

        // Assert: Debug formats without panic
        let debug = format!("{:?}", sg_no_tap);
        assert!(debug.contains("detect_layer"));

        // Act: with q_tap
        let sg_with_tap = mega_kernel_abi::SgConfig {
            detect_layer: 16,
            inject_offset: 0x4000,
            detect_offset: 0x5000,
            q_tap: Some(graph::QTapGraphConfig {
                sink_ptr: 0,
                step_index_ptr: 0,
                dtype: DType::F32,
                position: graph::QTapPosition::LastToken,
                num_slots: 4,
            }),
        };

        // Assert: q_tap is present
        assert!(sg_with_tap.q_tap.is_some());
        assert_eq!(sg_with_tap.q_tap.as_ref().unwrap().num_slots, 4);
    }

    /// Verify InferenceCompiler::print_resource_report does not panic in any
    /// environment (whether or not GLLM_DEBUG_RESOURCE is set).
    #[test]
    fn test_print_resource_report_no_panic() {
        // Arrange
        let compiler = InferenceCompiler::new();

        // Act & Assert: must not panic
        compiler.print_resource_report();
    }
}
