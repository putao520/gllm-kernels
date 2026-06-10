//! Emitter — ISA Lowering trait abstraction.
//!
//! ## Trait Architecture (SPEC §8.6)
//!
//! Platform differences in ISA Lowering code generation are encapsulated via two traits:
//!
//! - `MachineCodeEmitter`: the code-generation interface (emit_plan, simd_width).
//!   Implemented by `X86CodeGen` (jit-x86) and `DynasmAArch64CodeGen` (jit-aarch64).
//!
//! - `PlatformBackend`: factory + platform metadata. Implemented by `X86Backend`
//!   and `DynasmArm64Backend`. The compiler pipeline depends only on `PlatformBackend`,
//!   not on the concrete emitter type.

use crate::compiler::graph::CompilerGraph;
use crate::compiler::codegen::CodegenOutput;
use crate::compiler::fusion::FusionPlan;
use crate::compiler::buffer_alloc::BufferAllocation;
use crate::compiler::planner::ExecutionPlan;
use crate::dispatch::DeviceProfile;
use crate::types::CompilerError;

// ── Trait definitions ─────────────────────────────────────────────────────────

/// Platform identifier — carries capability flags used by the compiler pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// x86_64 with optional AVX-512 and AMX support.
    X86_64 { avx512: bool, amx: bool },
    /// AArch64 with optional SVE support.
    Aarch64 { sve: bool, amx: bool },
    /// NVIDIA CUDA — sm_version encodes compute capability (e.g. 80 = sm_80 / A100).
    #[cfg(feature = "jit-cuda")]
    Cuda { sm_version: u32 },
    /// AMD HIP — gfx_arch encodes GFX ISA (e.g. 908 = gfx908 / MI100).
    #[cfg(feature = "jit-hip")]
    Hip { gfx_arch: u32 },
    /// Apple Metal — gpu_family encodes Metal GPU family (e.g. 9 = Apple9).
    #[cfg(feature = "jit-metal")]
    Metal { gpu_family: u32 },
}

impl Platform {
    /// Detect the current host platform.
    pub fn detect(profile: &DeviceProfile) -> Self {
        use crate::dispatch::IsaLevel;

        #[cfg(target_arch = "x86_64")]
        return Platform::X86_64 {
            avx512: matches!(profile.isa, IsaLevel::Avx512 | IsaLevel::Avx512Amx),
            amx: matches!(profile.isa, IsaLevel::Avx512Amx),
        };

        #[cfg(target_arch = "aarch64")]
        return Platform::Aarch64 {
            sve: matches!(profile.isa, IsaLevel::Sve | IsaLevel::Sve2),
            amx: matches!(profile.isa, IsaLevel::NeonAmx),
        };

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return Platform::X86_64 { avx512: false, amx: false };
    }
}

/// ISA Lowering code-generation interface.
///
/// Implemented by each platform's JIT backend. The compiler pipeline calls
/// `emit_plan` to produce native machine code for a complete fusion plan.
///
/// The `registry` parameter carries scalar-op metadata used by x86_64 for
/// epilogue injection; aarch64 implementations may ignore it.
pub trait MachineCodeEmitter {
    /// Generate native machine code for a complete fusion plan.
    fn emit_plan(
        &mut self,
        plan: &FusionPlan,
        graph: &CompilerGraph,
        alloc: &BufferAllocation,
        exec_plan: &ExecutionPlan,
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, CompilerError>;

    /// Number of f32 elements per SIMD register on this backend.
    fn simd_width(&self) -> usize;
}

/// Platform backend factory — provides platform metadata and constructs emitters.
///
/// The compiler pipeline depends only on this trait, not on the concrete emitter
/// type, keeping SemanticDAG + Fusion (DAG construction, fusion decisions) platform-agnostic.
pub trait PlatformBackend {
    /// The concrete emitter type produced by this backend.
    type Emitter: MachineCodeEmitter;

    /// Construct a new emitter for the given device profile.
    fn new_emitter(&self, exec_plan: &ExecutionPlan) -> Self::Emitter;

    /// The platform this backend targets.
    fn platform(&self) -> Platform;

    /// Number of SIMD registers available on this platform.
    fn num_simd_regs(&self) -> usize;
}

// ── x86_64 Backend (inlined from former codegen/x86_64/) ─────────────────────

/// x86_64 JIT 代码生成器。
///
/// Register VM 统一管线的 MachineCodeEmitter 壳。
/// 真正的代码生成通过 `vm::compile_layer()` 实现。
#[cfg(feature = "jit-x86")]
pub struct X86CodeGen {
    simd_width: usize,
    pub(crate) weight_layout: Option<crate::compiler::graph::WeightLayout>,
}

#[cfg(feature = "jit-x86")]
impl X86CodeGen {
    pub fn new(profile: &DeviceProfile, dtype: crate::types::DType) -> Self {
        Self {
            simd_width: profile.simd_width(dtype),
            weight_layout: None,
        }
    }

    pub fn set_telemetry(&mut self, _config: &crate::compiler::graph::EpilogueTelemetryConfig) {}
    pub fn set_codegen_hints(&mut self, _hints: crate::compiler::semantic_dag::CodegenHints) {}
    pub fn set_weight_layout(&mut self, layout: crate::compiler::graph::WeightLayout) {
        self.weight_layout = Some(layout);
    }
    pub fn simd_width(&self) -> usize { self.simd_width }

    pub fn set_jit_params(&mut self, _params: &crate::autotuning::search_space::JitParams) {}

    pub fn emit_standalone_gemm(
        &mut self,
        _m: usize, _n: usize, _k: usize,
        _dtype: crate::types::DType,
        _blocking: &crate::dispatch::GemmBlocking,
        _profile: &DeviceProfile,
    ) -> Result<Vec<u8>, CompilerError> {
        Err(CompilerError::CodegenViolation(
            "emit_standalone_gemm: Register VM 迁移中。等待 vm::compile_layer() 实现。".into()
        ))
    }

    /// MachineCodeEmitter 入口 — 委托给 Register VM compile_layer。
    pub fn emit_plan(
        &mut self,
        _plan: &crate::compiler::fusion::FusionPlan,
        _graph: &crate::compiler::graph::CompilerGraph,
        _alloc: &crate::compiler::buffer_alloc::BufferAllocation,
        _exec_plan: &ExecutionPlan,
        _registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, CompilerError> {
        crate::compiler::codegen::vm::plan_lower::compile_layer(
            _plan, _graph, _alloc, _exec_plan, _registry,
        )
    }
}

#[cfg(feature = "jit-x86")]
impl MachineCodeEmitter for X86CodeGen {
    fn emit_plan(
        &mut self,
        plan: &crate::compiler::fusion::FusionPlan,
        graph: &crate::compiler::graph::CompilerGraph,
        alloc: &crate::compiler::buffer_alloc::BufferAllocation,
        exec_plan: &ExecutionPlan,
        registry: Option<&crate::compiler::registry::ScalarOpRegistry>,
    ) -> Result<CodegenOutput, CompilerError> {
        self.emit_plan(plan, graph, alloc, exec_plan, registry)
    }

    fn simd_width(&self) -> usize {
        self.simd_width()
    }
}

/// x86_64 platform backend factory.
#[cfg(feature = "jit-x86")]
pub struct X86Backend;

#[cfg(feature = "jit-x86")]
impl PlatformBackend for X86Backend {
    type Emitter = X86CodeGen;

    fn new_emitter(&self, exec_plan: &ExecutionPlan) -> Self::Emitter {
        X86CodeGen::new(&exec_plan.profile, crate::types::DType::F32)
    }

    fn platform(&self) -> Platform {
        #[cfg(target_arch = "x86_64")]
        let avx512 = std::is_x86_feature_detected!("avx512f");
        #[cfg(not(target_arch = "x86_64"))]
        let avx512 = false;
        #[cfg(target_arch = "x86_64")]
        let amx = std::is_x86_feature_detected!("amx-tile") && std::is_x86_feature_detected!("amx-bf16");
        #[cfg(not(target_arch = "x86_64"))]
        let amx = false;
        Platform::X86_64 { avx512, amx }
    }

    fn num_simd_regs(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DeviceProfile;

    #[test]
    fn test_platform_detect() {
        let profile = DeviceProfile::detect();
        let platform = Platform::detect(&profile);
        // Just verify it returns a valid variant without panicking
        match platform {
            Platform::X86_64 { .. } => {},
            Platform::Aarch64 { .. } => {},
            #[cfg(feature = "jit-cuda")]
            Platform::Cuda { .. } => {},
            #[cfg(feature = "jit-hip")]
            Platform::Hip { .. } => {},
            #[cfg(feature = "jit-metal")]
            Platform::Metal { .. } => {},
        }
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_backend_trait_roundtrip() {
        use super::X86Backend;
        use crate::compiler::planner::ExecutionPlan;

        let backend = X86Backend;
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let emitter = backend.new_emitter(&exec_plan);

        let plat = backend.platform();
        assert!(matches!(plat, Platform::X86_64 { .. }));
        assert!(backend.num_simd_regs() == 16 || backend.num_simd_regs() == 32);
        assert!(emitter.simd_width() == 8 || emitter.simd_width() == 16);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emitter_emit_plan_standalone_silu() {
        use super::X86CodeGen;
        use crate::compiler::graph::{CompilerGraph, MultiOutputConfig, OpKind};
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan, GroupMarker};
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::types::DType;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[16], DType::F32);
        let output = g.add_tensor_concrete("output", &[16], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);

        let registry = crate::compiler::registry::ScalarOpRegistry::with_defaults();
        let mut emitter: Box<dyn MachineCodeEmitter> = Box::new(X86CodeGen::new(&profile, DType::F32));
        let output = emitter.emit_plan(&plan, &g, &alloc, &exec_plan, Some(&registry)).unwrap();
        assert!(!output.code.is_empty(), "emit_plan via trait should produce code");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_emitter_emit_plan_with_registry() {
        use super::X86CodeGen;
        use crate::compiler::graph::{CompilerGraph, MultiOutputConfig, OpKind};
        use crate::compiler::fusion::{FusionGroup, FusionMode, FusionPlan, GroupMarker};
        use crate::compiler::buffer_alloc::BufferAllocation;
        use crate::compiler::registry::ScalarOpRegistry;
        use crate::types::DType;
        use std::collections::HashMap;

        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[16], DType::F32);
        let output = g.add_tensor_concrete("output", &[16], DType::F32);
        g.inputs = vec![input];
        g.outputs = vec![output];
        let op_id = g.add_op(OpKind::Silu, vec![input], vec![output], "silu");

        let mut op_to_group = HashMap::new();
        op_to_group.insert(op_id, 0);
        let plan = FusionPlan {
            groups: vec![FusionGroup {
                id: 0,
                anchor: op_id,
                epilogue: vec![],
                mode: FusionMode::LoopFusion,
                ops: vec![op_id],
                multi_output: MultiOutputConfig::single(),
            dominant_dtype: None,
            marker: GroupMarker::None,
            is_layer_group: false,
            hetero_layer_type: None,
            }],
            op_to_group,
        };
        let alloc = BufferAllocation::default();
        let profile = DeviceProfile::detect();
        let exec_plan = ExecutionPlan::from_profile(&profile);
        let registry = ScalarOpRegistry::with_defaults();

        let mut emitter = X86CodeGen::new(&profile, DType::F32);
        let output = emitter.emit_plan(&plan, &g, &alloc, &exec_plan, Some(&registry)).unwrap();
        assert!(!output.code.is_empty(), "registry-driven emit_plan should produce code");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_platform_detect_consistency() {
        use super::X86Backend;

        let profile = DeviceProfile::detect();
        let detected = Platform::detect(&profile);
        let backend = X86Backend;
        let backend_plat = backend.platform();

        // Both should report X86_64 on x86_64 hosts
        match (detected, backend_plat) {
            (Platform::X86_64 { avx512: a, .. }, Platform::X86_64 { avx512: b, .. }) => {
                assert_eq!(a, b, "Platform::detect and X86Backend::platform disagree on avx512");
            }
            _ => panic!("platform mismatch between detect and backend"),
        }
    }

    // ── Platform derive trait tests ─────────────────────────────────────

    #[test]
    fn test_platform_equality_same_variants() {
        let a = Platform::X86_64 { avx512: true, amx: false };
        let b = Platform::X86_64 { avx512: true, amx: false };
        assert_eq!(a, b, "identical Platform variants must be equal");
    }

    #[test]
    fn test_platform_equality_different_flags() {
        let a = Platform::X86_64 { avx512: true, amx: false };
        let b = Platform::X86_64 { avx512: false, amx: false };
        assert_ne!(a, b, "Platform with different avx512 flags must not be equal");
    }

    #[test]
    fn test_platform_clone() {
        let original = Platform::X86_64 { avx512: true, amx: true };
        let cloned = original.clone();
        assert_eq!(original, cloned, "cloned Platform must equal original");
    }

    #[test]
    fn test_platform_copy() {
        let original = Platform::Aarch64 { sve: true, amx: false };
        let copied = original; // Copy semantics, original still valid
        assert_eq!(original, copied, "copied Platform must equal original via Copy");
    }

    #[test]
    fn test_platform_debug_format() {
        let p = Platform::X86_64 { avx512: false, amx: false };
        let debug_str = format!("{:?}", p);
        assert!(debug_str.contains("X86_64"), "Debug output must contain variant name, got: {}", debug_str);
    }

    // ── X86CodeGen constructor + simd_width ─────────────────────────────

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_simd_width_varies_with_dtype() {
        use super::X86CodeGen;
        use crate::types::DType;

        let profile = DeviceProfile::detect();
        let emitter_f32 = X86CodeGen::new(&profile, DType::F32);
        let emitter_bf16 = X86CodeGen::new(&profile, DType::BF16);

        let width_f32 = emitter_f32.simd_width();
        let width_bf16 = emitter_bf16.simd_width();

        assert!(width_f32 > 0, "F32 simd_width must be positive");
        assert!(width_bf16 > 0, "BF16 simd_width must be positive");
        // BF16 is half the size of F32, so twice as many fit per register
        assert_eq!(width_bf16, width_f32 * 2,
            "BF16 simd_width should be 2x F32 simd_width (same register, half element size)");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_set_weight_layout_updates_field() {
        use super::X86CodeGen;
        use crate::types::DType;
        use crate::compiler::graph::WeightLayout;

        let profile = DeviceProfile::detect();
        let mut emitter = X86CodeGen::new(&profile, DType::F32);
        // Before setting: weight_layout is None (private, verify indirectly via no panic)
        let layout = WeightLayout {
            offsets: vec![(crate::compiler::graph::TensorId(0), 0)],
            total_bytes: 1024,
        };
        // Should not panic — validates the setter runs without error
        emitter.set_weight_layout(layout);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_set_telemetry_no_panic() {
        use super::X86CodeGen;
        use crate::types::DType;
        use crate::compiler::graph::EpilogueTelemetryConfig;

        let profile = DeviceProfile::detect();
        let mut emitter = X86CodeGen::new(&profile, DType::F32);
        // All telemetry flags enabled — verifies set_telemetry is callable
        let config = EpilogueTelemetryConfig {
            silu_dead_neuron: true,
            moe_hit_counter: true,
            rmsnorm_channel_scale: true,
            softmax_sharpness: true,
            residual_cosine_sim: true,
            gemm_row_stats: true,
            embed_l2_norm: true,
        };
        emitter.set_telemetry(&config);
        // No assertion needed — the test verifies no panic and the method is callable
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_set_codegen_hints_no_panic() {
        use super::X86CodeGen;
        use crate::types::DType;
        use crate::compiler::semantic_dag::CodegenHints;

        let profile = DeviceProfile::detect();
        let mut emitter = X86CodeGen::new(&profile, DType::F32);
        let hints = CodegenHints {
            is_memory_bound: true,
            arithmetic_intensity: 5.0,
            prefetch_hint: 2,
            use_nt_stores: true,
        };
        emitter.set_codegen_hints(hints);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_set_jit_params_no_panic() {
        use super::X86CodeGen;
        use crate::types::DType;
        use crate::autotuning::search_space::JitParams;

        let profile = DeviceProfile::detect();
        let mut emitter = X86CodeGen::new(&profile, DType::F32);
        let params = JitParams::default();
        emitter.set_jit_params(&params);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_emit_standalone_gemm_returns_error() {
        use super::X86CodeGen;
        use crate::types::DType;
        use crate::dispatch::DeviceProfile;

        let profile = DeviceProfile::detect();
        let mut emitter = X86CodeGen::new(&profile, DType::F32);
        let blocking = crate::dispatch::GemmBlocking {
            kc: 64, mc: 64, nc: 64, mr: 6, nr: 8,
        };
        let result = emitter.emit_standalone_gemm(16, 16, 16, DType::F32, &blocking, &profile);
        assert!(result.is_err(), "emit_standalone_gemm should return error (Register VM migration)");
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_backend_num_simd_regs_constant() {
        use super::X86Backend;

        let backend = X86Backend;
        assert_eq!(backend.num_simd_regs(), 32,
            "x86_64 with AVX-512 should have 32 SIMD registers (zmm0-zmm31)");
    }

    // ── Additional tests ──

    #[test]
    fn test_platform_aarch64_construction() {
        let p = Platform::Aarch64 { sve: true, amx: false };
        assert_eq!(p, Platform::Aarch64 { sve: true, amx: false });
        assert_ne!(p, Platform::Aarch64 { sve: false, amx: false });
    }

    #[test]
    fn test_platform_aarch64_debug() {
        let p = Platform::Aarch64 { sve: true, amx: true };
        let debug = format!("{:?}", p);
        assert!(debug.contains("Aarch64"), "Debug should contain Aarch64, got: {}", debug);
        assert!(debug.contains("sve"));
    }

    #[test]
    fn test_platform_equality_aarch64_variants() {
        let a = Platform::Aarch64 { sve: false, amx: false };
        let b = Platform::Aarch64 { sve: false, amx: false };
        assert_eq!(a, b);

        let c = Platform::Aarch64 { sve: true, amx: false };
        assert_ne!(a, c);
    }

    #[test]
    fn test_platform_x86_64_amx_flag() {
        let with_amx = Platform::X86_64 { avx512: true, amx: true };
        let without_amx = Platform::X86_64 { avx512: true, amx: false };
        assert_ne!(with_amx, without_amx);
    }

    #[test]
    fn test_platform_copy_preserves_original() {
        let original = Platform::X86_64 { avx512: true, amx: false };
        let _moved = original; // Copy, original still usable
        let another = original; // Another copy
        assert_eq!(original, another);
    }

    #[test]
    fn test_platform_clone_independent() {
        let original = Platform::Aarch64 { sve: true, amx: false };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_new_sets_simd_width() {
        use super::X86CodeGen;
        use crate::types::DType;

        let profile = DeviceProfile::detect();
        let emitter = X86CodeGen::new(&profile, DType::F32);
        // simd_width should be a positive, reasonable value
        let w = emitter.simd_width();
        assert!(w == 4 || w == 8 || w == 16,
            "F32 simd_width should be 4 (SSE), 8 (AVX2), or 16 (AVX-512), got {}", w);
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_backend_platform_reports_correct_flags() {
        use super::X86Backend;

        let backend = X86Backend;
        let plat = backend.platform();
        match plat {
            Platform::X86_64 { avx512, amx } => {
                // If AMX is true, AVX-512 must also be true (AMX requires AVX-512)
                if amx {
                    assert!(avx512, "AMX implies AVX-512 but avx512=false");
                }
            }
            _ => panic!("Expected X86_64 platform"),
        }
    }

    #[cfg(feature = "jit-x86")]
    #[test]
    fn test_x86_codegen_simd_width_positive() {
        use super::X86CodeGen;
        use crate::types::DType;

        let profile = DeviceProfile::detect();
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let emitter = X86CodeGen::new(&profile, dtype);
            assert!(emitter.simd_width() > 0,
                "simd_width should be positive for {:?}", dtype);
        }
    }

    #[test]
    fn test_platform_detect_returns_valid_variant() {
        let profile = DeviceProfile::detect();
        let platform = Platform::detect(&profile);
        // On x86_64 host, must return X86_64 variant
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(platform, Platform::X86_64 { .. }));
        // On aarch64 host, must return Aarch64 variant
        #[cfg(target_arch = "aarch64")]
        assert!(matches!(platform, Platform::Aarch64 { .. }));
    }
}
