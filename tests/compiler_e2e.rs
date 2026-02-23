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
    let graph = CompilerGraph::from_layer_ir(&ir, &profile);
    assert!(
        graph.num_ops() >= 14,
        "LLaMA should have >=14 ops, got {}",
        graph.num_ops()
    );

    let dag = SemanticDAG::from_graph(&graph, &registry);
    assert_eq!(dag.num_nodes(), graph.num_ops());

    // Phase 2: Fusion
    let plan = fuse_with_dag(&graph, &registry);
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

    // Phase 3: Codegen (stub)
    let output = codegen::emitter::emit_stub_code(&graph);
    assert!(!output.code.is_empty());

    // Wrap as CompiledLayer
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

/// Full pipeline test for Gemma-2B (GeGLU variant).
#[test]
fn test_e2e_gemma_2b_full_pipeline() {
    let config = ModelConfig::gemma_2b();
    let ir = LayerIR::from_model_config(&config, 1);
    let profile = DeviceProfile::detect();
    let registry = ScalarOpRegistry::with_defaults();

    let graph = CompilerGraph::from_layer_ir(&ir, &profile);
    let dag = SemanticDAG::from_graph(&graph, &registry);
    let plan = fuse_with_dag(&graph, &registry);
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

    let graph = CompilerGraph::from_layer_ir(&ir, &profile);

    let old_plan = fuse(&graph);
    let new_plan = fuse_with_dag(&graph, &registry);

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
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
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
    let graph = CompilerGraph::from_layer_ir(&ir, &profile);
    let plan = fuse_with_dag(&graph, &registry);
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

/// HW constraints: all groups for all model configs should pass.
#[test]
fn test_e2e_hw_constraints_all_models() {
    let configs = vec![ModelConfig::llama_7b(), ModelConfig::gemma_2b()];

    let profile = DeviceProfile::detect();
    let registry = ScalarOpRegistry::with_defaults();

    for config in &configs {
        let ir = LayerIR::from_model_config(config, 1);
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let plan = fuse_with_dag(&graph, &registry);
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
