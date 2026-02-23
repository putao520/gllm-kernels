//! Layer 3: Inference Compiler — JIT compilation of transformer layers.
//!
//! The compiler takes a `ModelConfig`, builds a `LayerIR` for each layer,
//! plans execution via `ExecutionPlan`, generates machine code, and caches
//! the result.
//!
//! # Pipeline
//!
//! ```text
//! LayerIR → CompilerGraph → Phase 0 (ScalarOpRegistry + OpTrace via SymbolicExecutor)
//!         → Phase 1 (SemanticDAG: OpClass auto-derivation from OpTrace::ComputePattern)
//!         → Phase 2 (Fusion: fuse_with_dag + HW constraints + Parallel strategy + Buffer alloc)
//!         → Phase 3 (CodeGen: x86_64/aarch64 JIT via iced-x86/dynasm-rs)
//!         → CompiledLayer (cached)
//! ```
//!
//! Phase 0 extracts `OpTrace` from `extern "C"` scalar functions via binary
//! symbolic execution. Phase 1 builds a `SemanticDAG` with auto-derived
//! `OpClass`. Phase 2 performs fusion decisions with HW constraint checks,
//! parallel strategy selection, and interval-graph buffer allocation.
//! Phase 3 generates native machine code from `TraceOp` sequences.

pub mod ir;
pub mod planner;
pub mod executable;
pub mod cache;
pub mod codegen;
pub mod graph;
pub mod semantics;
pub mod fusion;
pub mod trace;
pub mod registry;
pub mod semantic_dag;
pub mod symexec;
pub mod hw_constraints;
pub mod parallel;
pub mod buffer_alloc;

pub use ir::{LayerIR, LayerArch};
pub use planner::{ExecutionPlan, FusionDecision, GemmShape, MicrokernelChoice};
pub use executable::{CompiledLayer, CompiledLayerFn};
pub use cache::{CompilationCache, CacheSource, IncrementalCompileResult, CACHE_VERSION};
pub use codegen::CodegenOutput;
pub use graph::{CompilerGraph, CompilerOp, OpKind, TensorId, OpId};
pub use semantics::OpSemantics;
pub use fusion::{FusionPlan, FusionGroup, FusionPattern};
pub use codegen::emitter::ScratchpadLayout;
pub use trace::{OpTrace, ComputePattern, TraceOp, ScalarFnSignature, ScalarParam};
pub use registry::{ScalarOpRegistry, OpKindKey};
pub use semantic_dag::{SemanticDAG, SemanticNode, OpClass, Bottleneck, TensorEdge};
pub use hw_constraints::{HwConstraintResult, HwConstraintChecker, ConstraintViolation};
pub use parallel::{ParallelStrategy, ParallelDim};
pub use buffer_alloc::{BufferAllocation, BufferSlot, Lifetime};

use crate::dispatch::{DeviceProfile, device_profile};
use crate::inference::types::{InferenceError, ModelConfig};

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
            cache: CompilationCache::new(),
        }
    }

    /// Compile all layers for a model. Returns one `CompiledLayer` per layer.
    ///
    /// Layers with identical shapes share the same compiled code (common case:
    /// all decoder layers are identical except for weight pointers).
    pub fn compile_model(
        &mut self,
        config: &ModelConfig,
        max_batch: usize,
    ) -> Result<Vec<CompiledLayer>, InferenceError> {
        let ir = LayerIR::from_model_config(config, max_batch);
        let hash = self.compute_hash(&ir);

        // Check cache
        if let Some(_cached) = self.cache.get(hash) {
            let mut layers = Vec::with_capacity(config.num_layers);
            for _ in 0..config.num_layers {
                let layer = self.cache.get(hash).ok_or_else(|| {
                    InferenceError::CompileError("cache inconsistency".into())
                })?;
                layers.push(layer);
            }
            return Ok(layers);
        }

        // Full JIT pipeline: LayerIR → Graph → Fuse → Emit → Code
        let output = self.jit_compile(&ir)?;

        // Cache the compiled code
        self.cache.put(hash, &output.code, output.scratchpad_bytes);

        // Create CompiledLayer instances for each layer
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let layer = CompiledLayer::from_code(&output.code, output.scratchpad_bytes, hash)?;
            layers.push(layer);
        }

        Ok(layers)
    }

    /// Compile a single layer (for testing or incremental compilation).
    pub fn compile_layer(
        &mut self,
        ir: &LayerIR,
    ) -> Result<CompiledLayer, InferenceError> {
        let hash = self.compute_hash(ir);

        if let Some(cached) = self.cache.get(hash) {
            return Ok(cached);
        }

        let output = self.jit_compile(ir)?;

        self.cache.put(hash, &output.code, output.scratchpad_bytes);

        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, hash)
    }

    /// Compile a model incrementally — only recompile layers whose hash
    /// is not already in the cache (memory or disk). Returns per-layer
    /// compiled code plus hit/miss statistics for logging.
    pub fn compile_model_incremental(
        &mut self,
        config: &ModelConfig,
        max_batch: usize,
    ) -> Result<IncrementalCompileResult, InferenceError> {
        let ir = LayerIR::from_model_config(config, max_batch);
        let hash = self.compute_hash(&ir);

        let mut memory_hits: usize = 0;
        let mut disk_hits: usize = 0;
        let mut compiled: usize = 0;

        // All decoder layers share the same IR shape, so one lookup decides.
        match self.cache.lookup(hash) {
            Some((_, CacheSource::Memory)) => {
                memory_hits = config.num_layers;
            }
            Some((_, CacheSource::Disk)) => {
                // First hit loaded from disk (now promoted to memory).
                disk_hits = 1;
                memory_hits = config.num_layers.saturating_sub(1);
            }
            None => {
                let output = self.jit_compile(&ir)?;
                self.cache.put(hash, &output.code, output.scratchpad_bytes);
                compiled = config.num_layers;
            }
        }

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let layer = self.cache.get(hash).ok_or_else(|| {
                InferenceError::CompileError("cache inconsistency after compilation".into())
            })?;
            layers.push(layer);
        }

        Ok(IncrementalCompileResult {
            layers,
            memory_hits,
            disk_hits,
            compiled,
        })
    }

    /// Clear the compilation cache (memory + disk).
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

    fn compute_hash(&self, ir: &LayerIR) -> u64 {
        let ir_desc = format!(
            "{:?}_h{}_nh{}_nkv{}_hd{}_inter{}_q{:?}_dt{:?}_mb{}_ms{}",
            ir.arch, ir.hidden, ir.num_heads, ir.num_kv_heads,
            ir.head_dim, ir.intermediate, ir.quant, ir.dtype,
            ir.max_batch, ir.max_seq,
        );
        let hw_fp = self.profile.hw_info.fingerprint();
        cache::config_hash(ir_desc.as_bytes(), &hw_fp)
    }

    /// Full JIT compilation pipeline:
    /// LayerIR → CompilerGraph → Phase 0 (registry) → Phase 1 (SemanticDAG)
    ///         → Phase 2 (fusion + HW + parallel + buffer) → Phase 3 (codegen)
    ///         → CodegenOutput
    ///
    /// Phase 3 has an MVP implementation under the `jit-x86` feature flag
    /// (see `codegen::x86_64::jit::X86CodeGen`). Without the feature flag,
    /// a stub is emitted as fallback.
    fn jit_compile(&self, ir: &LayerIR) -> Result<codegen::CodegenOutput, InferenceError> {
        // Phase 1: Build CompilerGraph DAG
        let graph = CompilerGraph::from_layer_ir(ir, &self.profile);

        // Phase 0 + 1: ScalarOpRegistry (OpTrace cache) + SemanticDAG (OpClass auto-derivation)
        let registry = ScalarOpRegistry::with_defaults();
        let _semantic_dag = SemanticDAG::from_graph(&graph, &registry);

        // Phase 2: Fusion decisions + buffer allocation
        let fusion_plan = fusion::fuse_with_dag(&graph, &registry);
        let lifetimes = buffer_alloc::analyze_lifetimes(&graph, &fusion_plan);
        let alloc = buffer_alloc::allocate_buffers(&lifetimes);

        // Phase 3: Code generation
        #[cfg(feature = "jit-x86")]
        {
            let mut cg = codegen::x86_64::X86CodeGen::new(&self.profile);
            return cg.emit_plan(&fusion_plan, &graph, &alloc, &self.profile, Some(&registry))
                .map_err(|e| InferenceError::CompileError(e));
        }

        #[cfg(not(feature = "jit-x86"))]
        {
            let _ = (fusion_plan, alloc);
            Ok(codegen::emitter::emit_stub_code(&graph))
        }
    }

    /// Compile a CompilerGraph directly.
    ///
    /// This is the primary entry point for GLLM integration: GLLM expands
    /// its high-level FusedGraph into an atomic-op DAG (`CompilerGraph`)
    /// and passes it here for JIT compilation.
    pub fn compile_graph(
        &mut self,
        graph: &CompilerGraph,
    ) -> Result<CompiledLayer, InferenceError> {
        let registry = ScalarOpRegistry::with_defaults();
        let fusion_plan = fusion::fuse_with_dag(graph, &registry);
        let lifetimes = buffer_alloc::analyze_lifetimes(graph, &fusion_plan);
        let alloc = buffer_alloc::allocate_buffers(&lifetimes);

        #[cfg(feature = "jit-x86")]
        let output = {
            let mut cg = codegen::x86_64::X86CodeGen::new(&self.profile);
            cg.emit_plan(&fusion_plan, graph, &alloc, &self.profile, Some(&registry))
                .map_err(|e| InferenceError::CompileError(e))?
        };

        #[cfg(not(feature = "jit-x86"))]
        let output = {
            let _ = (fusion_plan, alloc);
            codegen::emitter::emit_stub_code(graph)
        };

        let hash = 0; // TODO: graph-level content hash
        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::ModelConfig;

    #[test]
    fn test_compiler_new() {
        let compiler = InferenceCompiler::new();
        assert_eq!(compiler.cache_size(), 0);
    }

    #[test]
    fn test_compile_layer() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let mut compiler = InferenceCompiler::new();

        let layer = compiler.compile_layer(&ir).unwrap();
        assert!(layer.code_size() > 0);
        assert!(layer.scratchpad_bytes > 0);
        assert_eq!(compiler.cache_size(), 1);

        // Second compile should hit cache
        let layer2 = compiler.compile_layer(&ir).unwrap();
        assert_eq!(layer.config_hash, layer2.config_hash);
    }

    #[test]
    fn test_compile_model() {
        let mut config = ModelConfig::llama_7b();
        config.num_layers = 2;
        let mut compiler = InferenceCompiler::new();

        let layers = compiler.compile_model(&config, 1).unwrap();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].config_hash, layers[1].config_hash);
        assert!(layers[0].scratchpad_bytes > 0);
    }

    /// End-to-end: full JIT pipeline for LLaMA-7B decoder layer.
    #[test]
    fn test_e2e_llama_7b() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();

        // Phase 1: DAG
        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        assert!(graph.num_ops() >= 14);

        // Phase 2: Fusion
        let fplan = fusion::fuse(&graph);
        assert!(fplan.num_groups() < graph.num_ops());

        // Phase 3: Codegen (stub for now)
        let output = codegen::emitter::emit_stub_code(&graph);
        assert!(!output.code.is_empty());
        assert!(output.scratchpad_bytes > 0);

        // Wrap as CompiledLayer
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

    /// End-to-end: full JIT pipeline for Gemma-2B (GeGLU variant).
    #[test]
    fn test_e2e_gemma_2b() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();

        let graph = CompilerGraph::from_layer_ir(&ir, &profile);
        let fplan = fusion::fuse(&graph);
        let output = codegen::emitter::emit_stub_code(&graph);

        let mut compiler = InferenceCompiler::with_profile(profile);
        let layer = compiler.compile_layer(&ir).unwrap();
        assert!(layer.code_size() > 0);

        eprintln!(
            "E2E Gemma-2B: {} ops → {} groups → {} bytes code",
            graph.num_ops(),
            fplan.num_groups(),
            layer.code_size(),
        );
    }

    #[test]
    fn test_compile_model_incremental_cold() {
        let mut config = ModelConfig::llama_7b();
        config.num_layers = 3;
        let profile = DeviceProfile::detect();
        let mut compiler = InferenceCompiler::with_profile(profile);

        // First call: everything is a fresh compile
        let result = compiler.compile_model_incremental(&config, 1).unwrap();
        assert_eq!(result.layers.len(), 3);
        assert_eq!(result.compiled, 3);
        assert_eq!(result.memory_hits, 0);
        assert_eq!(result.disk_hits, 0);

        // Second call: all from memory
        let result2 = compiler.compile_model_incremental(&config, 1).unwrap();
        assert_eq!(result2.layers.len(), 3);
        assert_eq!(result2.compiled, 0);
        assert_eq!(result2.memory_hits, 3);
        assert_eq!(result2.disk_hits, 0);
    }
}
