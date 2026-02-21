//! Layer 3: Inference Compiler — JIT compilation of transformer layers.
//!
//! The compiler takes a `ModelConfig`, builds a `LayerIR` for each layer,
//! plans execution via `ExecutionPlan`, generates machine code via the
//! appropriate `LayerCodegen` backend, and caches the result.
//!
//! # Pipeline
//!
//! ```text
//! ModelConfig → LayerIR → ExecutionPlan → Codegen → CompiledLayer
//!                  ↑            ↑            ↑
//!              inference    dispatch     codegen/
//!              /types.rs   /device_     x86_64.rs
//!                          profile.rs   aarch64.rs
//! ```

pub mod ir;
pub mod planner;
pub mod executable;
pub mod cache;
pub mod codegen;

pub use ir::{LayerIR, LayerArch};
pub use planner::{ExecutionPlan, FusionDecision, GemmShape, MicrokernelChoice};
pub use executable::{CompiledLayer, CompiledLayerFn};
pub use cache::CompilationCache;
pub use codegen::{CodegenOutput, LayerCodegen};

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
        let plan = ExecutionPlan::build(&ir, &self.profile);

        // All layers share the same IR in standard transformers
        let hash = self.compute_hash(&ir);

        // Check cache
        if let Some(cached) = self.cache.get(hash) {
            // Replicate for all layers (they share the same code)
            let mut layers = Vec::with_capacity(config.num_layers);
            for _ in 0..config.num_layers {
                let layer = self.cache.get(hash).ok_or_else(|| {
                    InferenceError::CompileError("cache inconsistency".into())
                })?;
                layers.push(layer);
            }
            return Ok(layers);
        }

        // Compile
        let codegen = codegen::select_codegen(self.profile.isa)?;
        let output = codegen.generate(&ir, &plan)?;

        // Cache the compiled code
        self.cache.put(hash, &output.code, output.scratchpad_bytes);

        // Create CompiledLayer instances for each layer
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let layer = CompiledLayer::from_code(
                &output.code,
                output.scratchpad_bytes,
                hash,
            )?;
            layers.push(layer);
        }

        Ok(layers)
    }

    /// Compile a single layer (for testing or incremental compilation).
    pub fn compile_layer(
        &mut self,
        ir: &LayerIR,
    ) -> Result<CompiledLayer, InferenceError> {
        let plan = ExecutionPlan::build(ir, &self.profile);
        let hash = self.compute_hash(ir);

        if let Some(cached) = self.cache.get(hash) {
            return Ok(cached);
        }

        let codegen = codegen::select_codegen(self.profile.isa)?;
        let output = codegen.generate(ir, &plan)?;

        self.cache.put(hash, &output.code, output.scratchpad_bytes);

        CompiledLayer::from_code(&output.code, output.scratchpad_bytes, hash)
    }

    /// Clear the compilation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of cached compilations.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    fn compute_hash(&self, ir: &LayerIR) -> u64 {
        // Serialize IR fields into bytes for hashing
        let ir_desc = format!(
            "{:?}_h{}_nh{}_nkv{}_hd{}_inter{}_q{:?}_dt{:?}_mb{}_ms{}",
            ir.arch, ir.hidden, ir.num_heads, ir.num_kv_heads,
            ir.head_dim, ir.intermediate, ir.quant, ir.dtype,
            ir.max_batch, ir.max_seq,
        );
        let hw_fp = self.profile.hw_info.fingerprint();
        cache::config_hash(ir_desc.as_bytes(), &hw_fp)
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
        config.num_layers = 2; // small for testing
        let mut compiler = InferenceCompiler::new();

        let layers = compiler.compile_model(&config, 1).unwrap();
        assert_eq!(layers.len(), 2);
        // All layers should share the same config hash
        assert_eq!(layers[0].config_hash, layers[1].config_hash);
    }
}
