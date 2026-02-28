//! End-to-end compiler integration tests.
//!
//! Tests the full Phase 0-3 pipeline for various model architectures,
//! verifying correctness, consistency, and performance characteristics.
//!
//! Pipeline: LayerIR -> CompilerGraph -> Phase 0 (Registry+OpTrace)
//!         -> Phase 1 (SemanticDAG) -> Phase 2 (Fusion+HW+Parallel+Buffer)
//!         -> Phase 3 (Codegen) -> CompiledLayer

#![feature(f16)]

use gllm_kernels::compiler::buffer_alloc::{allocate_buffers, analyze_lifetimes};
use gllm_kernels::compiler::codegen;
use gllm_kernels::compiler::fusion::{fuse, fuse_with_dag};
use gllm_kernels::compiler::hw_constraints::check_plan;
use gllm_kernels::compiler::parallel::plan_parallelism;
use gllm_kernels::compiler::planner::ExecutionPlan;
use gllm_kernels::compiler::registry::ScalarOpRegistry;
use gllm_kernels::compiler::semantic_dag::SemanticDAG;
use gllm_kernels::compiler::{CompiledLayer, CompilerGraph, InferenceCompiler, LayerIR};
use gllm_kernels::dispatch::DeviceProfile;
use gllm_kernels::inference::types::ModelConfig;

// ── Full pipeline tests ──────────────────────────────────────────────

/// Full pipeline test for LLaMA-7B architecture.
#[test]
fn test_e2e_llama_7b_full_pipeline() {
    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();

    // Phase 0: Registry
    let registry = ScalarOpRegistry::with_defaults();

    // Phase 1: CompilerGraph + SemanticDAG
    let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
    assert!(
        graph.num_ops() >= 14,
        "LLaMA should have >=14 ops, got {}",
        graph.num_ops()
    );

    let dag = SemanticDAG::from_graph(&graph, &registry);
    assert_eq!(dag.num_nodes(), graph.num_ops());

    // Phase 2: Fusion
    let plan = fuse_with_dag(&graph, &registry, &profile);
    assert!(
        plan.num_groups() < graph.num_ops(),
        "Fusion should reduce groups: {} groups from {} ops",
        plan.num_groups(),
        graph.num_ops()
    );

    // Phase 2: HW constraints
    let hw_results = check_plan(&plan.groups, &graph, &profile);
    for result in &hw_results {
        assert!(
            result.valid,
            "Group {} failed HW constraints: {:?}",
            result.group_id, result.violations
        );
    }

    // Phase 2: Parallel strategy
    let parallel = plan_parallelism(&plan, &graph, &dag, &profile);
    assert_eq!(parallel.len(), plan.num_groups());

    // Phase 2: Buffer allocation
    let lifetimes = analyze_lifetimes(&graph, &plan);
    let alloc = allocate_buffers(&lifetimes);
    assert!(alloc.bytes_saved > 0, "Buffer coloring should save memory");

    // Phase 2: ExecutionPlan
    let exec_plan = ExecutionPlan::build(&ir, &profile);
    assert!(!exec_plan.fusions.is_empty());
    assert!(exec_plan.scratchpad_bytes > 0);

    // Phase 3: Codegen
    #[cfg(feature = "jit-x86")]
    {
        let mut cg = codegen::x86_64::X86CodeGen::new(&profile);
        let output = cg.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry))
            .expect("JIT codegen failed");
        let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, 0).unwrap();
        assert!(layer.code_size() > 0);
        eprintln!(
            "E2E LLaMA-7B: {} ops -> {} groups -> {} parallel strategies -> {} buffer bytes (saved {})",
            graph.num_ops(),
            plan.num_groups(),
            parallel.len(),
            alloc.total_bytes,
            alloc.bytes_saved
        );
    }
    #[cfg(not(feature = "jit-x86"))]
    {
        eprintln!(
            "E2E LLaMA-7B: {} ops -> {} groups -> {} parallel strategies -> {} buffer bytes (saved {}) [Phase 3 skipped]",
            graph.num_ops(),
            plan.num_groups(),
            parallel.len(),
            alloc.total_bytes,
            alloc.bytes_saved
        );
    }
}

/// Full pipeline test for Gemma-2B (GeGLU variant).
#[test]
fn test_e2e_gemma_2b_full_pipeline() {
    let config = ModelConfig::gemma_2b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let registry = ScalarOpRegistry::with_defaults();

    let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
    let dag = SemanticDAG::from_graph(&graph, &registry);
    let plan = fuse_with_dag(&graph, &registry, &profile);
    let hw_results = check_plan(&plan.groups, &graph, &profile);
    let parallel = plan_parallelism(&plan, &graph, &dag, &profile);
    let lifetimes = analyze_lifetimes(&graph, &plan);
    let alloc = allocate_buffers(&lifetimes);

    for result in &hw_results {
        assert!(
            result.valid,
            "Gemma group {} failed: {:?}",
            result.group_id, result.violations
        );
    }

    eprintln!(
        "E2E Gemma-2B: {} ops -> {} groups -> {} parallel strategies, {} buffer bytes saved",
        graph.num_ops(),
        plan.num_groups(),
        parallel.len(),
        alloc.bytes_saved
    );
}

// ── Fusion consistency ───────────────────────────────────────────────

/// Verify old fuse() and new fuse_with_dag() produce compatible results.
#[test]
fn test_e2e_fusion_consistency() {
    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let registry = ScalarOpRegistry::with_defaults();

    let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");

    let old_plan = fuse(&graph, &profile);
    let new_plan = fuse_with_dag(&graph, &registry, &profile);

    // New plan should have same or fewer groups (more aggressive fusion)
    let diff = (old_plan.num_groups() as i32 - new_plan.num_groups() as i32).abs();
    assert!(
        diff <= 3,
        "Fusion plans differ by {} groups (old={}, new={})",
        diff,
        old_plan.num_groups(),
        new_plan.num_groups()
    );

    // All ops should be covered
    let old_ops: usize = old_plan.groups.iter().map(|g| g.ops.len()).sum();
    let new_ops: usize = new_plan.groups.iter().map(|g| g.ops.len()).sum();
    assert_eq!(old_ops, graph.num_ops(), "Old plan missing ops");
    assert_eq!(new_ops, graph.num_ops(), "New plan missing ops");
}

// ── InferenceCompiler integration ────────────────────────────────────

/// InferenceCompiler end-to-end: compile + cache hit.
#[test]
fn test_e2e_inference_compiler() {
    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let mut compiler = InferenceCompiler::with_profile(profile);

    // First compile: cache miss
    let layer1 = compiler.compile_layer(&ir).unwrap();
    assert!(layer1.code_size() > 0);
    assert_eq!(compiler.cache_size(), 1);

    // Second compile: cache hit
    let layer2 = compiler.compile_layer(&ir).unwrap();
    assert_eq!(layer1.config_hash, layer2.config_hash);
    assert_eq!(compiler.cache_size(), 1); // no new entry
}

/// Incremental compilation: verify hit/miss stats.
#[test]
fn test_e2e_incremental_compilation() {
    let mut config = ModelConfig::llama_7b();
    config.num_layers = 4;
    let profile = DeviceProfile::detect();
    let mut compiler = InferenceCompiler::with_profile(profile);

    // Cold compile
    let result = compiler.compile_model_incremental(&config, 1).unwrap();
    assert_eq!(result.layers.len(), 4);
    assert_eq!(result.compiled, 4);
    assert_eq!(result.memory_hits, 0);

    // Warm compile
    let result2 = compiler.compile_model_incremental(&config, 1).unwrap();
    assert_eq!(result2.compiled, 0);
    assert_eq!(result2.memory_hits, 4);
}

// ── SemanticDAG coverage ─────────────────────────────────────────────

/// SemanticDAG consistency: every graph op has a DAG node.
#[test]
fn test_e2e_semantic_dag_coverage() {
    for config in &[ModelConfig::llama_7b(), ModelConfig::gemma_2b()] {
        let ir = LayerIR::from_model_config(config, 1);
        let profile = DeviceProfile::detect();
        let registry = ScalarOpRegistry::with_defaults();
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let dag = SemanticDAG::from_graph(&graph, &registry);

        assert_eq!(
            dag.num_nodes(),
            graph.num_ops(),
            "DAG nodes ({}) != graph ops ({})",
            dag.num_nodes(),
            graph.num_ops()
        );
    }
}

// ── Buffer allocation correctness ────────────────────────────────────

/// Buffer allocation: verify no overlapping live buffers.
#[test]
fn test_e2e_buffer_no_overlap() {
    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let registry = ScalarOpRegistry::with_defaults();
    let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
    let plan = fuse_with_dag(&graph, &registry, &profile);
    let lifetimes = analyze_lifetimes(&graph, &plan);
    let alloc = allocate_buffers(&lifetimes);

    // Check no two simultaneously-live buffers overlap in memory
    for (i, lt_i) in lifetimes.iter().enumerate() {
        let slot_i = alloc.slots.iter().find(|s| s.tensor_id == lt_i.tensor_id);
        if slot_i.is_none() {
            continue;
        }
        let si = slot_i.unwrap();

        for (j, lt_j) in lifetimes.iter().enumerate() {
            if j <= i {
                continue;
            }
            let slot_j = alloc.slots.iter().find(|s| s.tensor_id == lt_j.tensor_id);
            if slot_j.is_none() {
                continue;
            }
            let sj = slot_j.unwrap();

            // Check if lifetimes overlap
            let lives_overlap =
                lt_i.first_use <= lt_j.last_use && lt_j.first_use <= lt_i.last_use;
            if lives_overlap {
                // Memory ranges must not overlap
                let mem_overlap =
                    si.offset < sj.offset + sj.size_bytes && sj.offset < si.offset + si.size_bytes;
                assert!(
                    !mem_overlap,
                    "Buffers {} and {} overlap in both lifetime and memory!",
                    lt_i.tensor_id.0, lt_j.tensor_id.0
                );
            }
        }
    }
}

// ── HW constraints across models ─────────────────────────────────────

// ── Concurrent compilation ─────────────────────────────────────────

/// Spawn 4 threads, each compiling a different LayerIR (different model
/// configs / hidden sizes). Verify all succeed and produce valid code.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_concurrent_compile_different_layers() {
    use std::sync::Arc;
    use std::thread;

    let configs: Vec<ModelConfig> = vec![
        ModelConfig::llama_7b(),
        ModelConfig::gemma_2b(),
        ModelConfig::mistral_7b(),
        ModelConfig::phi_2b(),
    ];

    let profile = Arc::new(DeviceProfile::detect());

    let handles: Vec<_> = configs
        .into_iter()
        .map(|config| {
            let profile = Arc::clone(&profile);
            thread::spawn(move || {
                let ir = LayerIR::from_model_config(&config, 1);
                let mut compiler = InferenceCompiler::with_profile((*profile).clone());
                let layer = compiler.compile_layer(&ir).unwrap();
                assert!(layer.code_size() > 0);
                assert!(layer.scratchpad_bytes > 0);
                (config.arch, layer.config_hash, layer.code_size())
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All 4 threads produced valid, distinct compilations
    assert_eq!(results.len(), 4);
    for (i, (arch_i, hash_i, _)) in results.iter().enumerate() {
        for (j, (_, hash_j, _)) in results.iter().enumerate() {
            if i != j {
                assert_ne!(
                    hash_i, hash_j,
                    "Different models should produce different hashes"
                );
            }
        }
    }

    eprintln!(
        "Concurrent different-layer compile: {:?}",
        results.iter().map(|(a, _, sz)| format!("{a:?}={sz}B")).collect::<Vec<_>>()
    );
}

/// Spawn 4 threads all compiling the same LayerIR through a shared
/// compiler (behind Arc<Mutex>). Verify all produce identical results
/// and the cache is hit after the first compilation.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_concurrent_compile_same_layer() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let compiler = Arc::new(Mutex::new(InferenceCompiler::with_profile(profile)));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let compiler = Arc::clone(&compiler);
            let ir = ir.clone();
            thread::spawn(move || {
                let mut comp = compiler.lock().unwrap();
                let layer = comp.compile_layer(&ir).unwrap();
                (layer.config_hash, layer.code_size(), layer.scratchpad_bytes)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should get identical results
    let (first_hash, first_size, first_scratch) = results[0];
    for (hash, size, scratch) in &results[1..] {
        assert_eq!(*hash, first_hash, "Cache should return same hash");
        assert_eq!(*size, first_size, "Cache should return same code size");
        assert_eq!(*scratch, first_scratch, "Cache should return same scratchpad");
    }

    // Only 1 entry in cache (all threads compiled the same IR)
    let comp = compiler.lock().unwrap();
    assert_eq!(comp.cache_size(), 1);
}

/// Pre-compile a layer, then spawn 4 threads that all call compile_layer()
/// with the same IR. All should hit the cache and return consistent results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_compile_cache_thread_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let config = ModelConfig::llama_7b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let mut compiler = InferenceCompiler::with_profile(profile);

    // Pre-compile to populate cache
    let first = compiler.compile_layer(&ir).unwrap();
    assert_eq!(compiler.cache_size(), 1);

    let compiler = Arc::new(Mutex::new(compiler));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let compiler = Arc::clone(&compiler);
            let ir = ir.clone();
            thread::spawn(move || {
                let mut comp = compiler.lock().unwrap();
                let layer = comp.compile_layer(&ir).unwrap();
                (layer.config_hash, layer.code_size(), layer.scratchpad_bytes)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All cache hits should match the pre-compiled result
    for (hash, size, scratch) in &results {
        assert_eq!(*hash, first.config_hash);
        assert_eq!(*size, first.code_size());
        assert_eq!(*scratch, first.scratchpad_bytes);
    }

    // Cache size unchanged — no new entries
    let comp = compiler.lock().unwrap();
    assert_eq!(comp.cache_size(), 1);
}

/// Verify concurrent creation of ScalarOpRegistry is safe and produces
/// consistent results across threads.
#[test]
fn test_concurrent_scalar_op_registry() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let reg = ScalarOpRegistry::with_defaults();
                let num_entries = reg.num_entries();
                let num_traces = reg.num_traces();

                // Spot-check a few traces exist
                assert!(reg.get_trace(&gllm_kernels::compiler::OpKindKey::Silu).is_some());
                assert!(reg.get_trace(&gllm_kernels::compiler::OpKindKey::Gelu).is_some());
                assert!(reg.get_trace(&gllm_kernels::compiler::OpKindKey::Gemm).is_some());

                (num_entries, num_traces)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should see the same registry contents
    let (first_entries, first_traces) = results[0];
    assert!(first_entries > 0);
    assert!(first_traces > 0);
    for (entries, traces) in &results[1..] {
        assert_eq!(*entries, first_entries, "Registry entry count mismatch across threads");
        assert_eq!(*traces, first_traces, "Registry trace count mismatch across threads");
    }
}

/// Compile the same IR in 4 independent threads (each with its own compiler),
/// verify all produce byte-identical code.
#[test]
#[cfg(feature = "jit-x86")]
#[cfg(target_arch = "x86_64")]
fn test_deterministic_across_threads() {
    use std::sync::Arc;
    use std::thread;

    let config = ModelConfig::llama_7b();
    let ir = Arc::new(LayerIR::from_model_config(&config, 1));
    let profile = Arc::new(DeviceProfile::detect());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let ir = Arc::clone(&ir);
            let profile = Arc::clone(&profile);
            thread::spawn(move || {
                let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
                let registry = ScalarOpRegistry::with_defaults();
                let plan = fuse_with_dag(&graph, &registry, &profile);
                let lifetimes = analyze_lifetimes(&graph, &plan);
                let alloc = allocate_buffers(&lifetimes);
                let mut cg = codegen::x86_64::X86CodeGen::new(&profile);
                let output = cg.emit_plan(&plan, &graph, &alloc, &profile, Some(&registry))
                    .expect("JIT codegen failed");
                (output.code, output.scratchpad_bytes)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let (ref first_code, first_scratch) = results[0];
    assert!(!first_code.is_empty());
    for (i, (code, scratch)) in results.iter().enumerate().skip(1) {
        assert_eq!(
            code, first_code,
            "Thread {} produced different code bytes than thread 0",
            i
        );
        assert_eq!(
            *scratch, first_scratch,
            "Thread {} produced different scratchpad size than thread 0",
            i
        );
    }

    eprintln!(
        "Determinism verified: {} bytes identical across 4 threads",
        first_code.len()
    );
}

// ── HW constraints across models ─────────────────────────────────────

/// HW constraints: all groups for all model configs should pass.
#[test]
fn test_e2e_hw_constraints_all_models() {
    let configs = vec![ModelConfig::llama_7b(), ModelConfig::gemma_2b()];

    let profile = DeviceProfile::detect();
    let registry = ScalarOpRegistry::with_defaults();

    for config in &configs {
        let ir = LayerIR::from_model_config(config, 1);
        let graph = CompilerGraph::from_layer_ir(&ir, &profile).expect("from_layer_ir failed");
        let plan = fuse_with_dag(&graph, &registry, &profile);
        let results = check_plan(&plan.groups, &graph, &profile);

        let violations: Vec<_> = results.iter().filter(|r| !r.valid).collect();
        assert!(
            violations.is_empty(),
            "Model {:?} has {} HW constraint violations",
            config.arch,
            violations.len()
        );
    }
}
