//! Acceleration Registry — 声明式加速指令布局注册表 (SPEC §3.10 §8.1).
//!
//! 每个加速指令声明: 硬件要求、算子模式、输入/输出布局约束、收益函数。
//! LayoutNegotiator (R1.5) 消费此注册表进行流水线级联布局协商。

use crate::compiler::semantic_dag::OpClass;
use crate::compiler::pain_point::BottleneckType;
use crate::dispatch::device_profile::{DeviceProfile, IsaLevel};

/// Unique identifier for an acceleration feature.
pub type AccelerationId = &'static str;

/// Hardware requirement for an acceleration feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareRequirement {
    /// x86_64 with AVX-512 (any CPU with AVX-512 F/VL/BW/DQ).
    Avx512,
    /// x86_64 with AVX-512 + Intel AMX tile instructions.
    Avx512Amx,
    /// x86_64 with AVX2 (haswell+).
    Avx2,
    /// AArch64 with NEON (any ARMv8+).
    Neon,
    /// AArch64 with SVE (any VL).
    Sve,
    /// AArch64 with SVE2.
    Sve2,
    /// AArch64 with Apple AMX (M1+).
    NeonAmx,
    /// GPU NVIDIA SM80+ (A100).
    GpuSm80,
    /// GPU NVIDIA SM90+ (H100).
    GpuSm90,
    /// GPU NVIDIA SM100+ (next gen).
    GpuSm100,
    /// No specific hardware requirement — always available.
    Any,
}

impl HardwareRequirement {
    /// Check if this requirement is satisfied by the given device profile.
    pub fn satisfied_by(&self, device: &DeviceProfile) -> bool {
        match self {
            HardwareRequirement::Avx512 => matches!(device.isa, IsaLevel::Avx512 | IsaLevel::Avx512Amx),
            HardwareRequirement::Avx512Amx => matches!(device.isa, IsaLevel::Avx512Amx),
            HardwareRequirement::Avx2 => matches!(device.isa, IsaLevel::Avx2 | IsaLevel::Avx512 | IsaLevel::Avx512Amx),
            HardwareRequirement::Neon => matches!(device.isa, IsaLevel::Neon | IsaLevel::Sve | IsaLevel::Sve2 | IsaLevel::NeonAmx),
            HardwareRequirement::Sve => matches!(device.isa, IsaLevel::Sve | IsaLevel::Sve2),
            HardwareRequirement::Sve2 => matches!(device.isa, IsaLevel::Sve2),
            HardwareRequirement::NeonAmx => matches!(device.isa, IsaLevel::NeonAmx),
            HardwareRequirement::GpuSm80 | HardwareRequirement::GpuSm90 | HardwareRequirement::GpuSm100 => {
                // GPU requirements checked via GPU backend detection.
                // For CPU-only builds, these are never satisfied.
                false
            }
            HardwareRequirement::Any => true,
        }
    }
}

/// Layout constraint — a set of acceptable data layouts.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutConstraint {
    /// Standard row-major with alignment.
    RowMajor { align_bytes: usize },
    /// Column-major with alignment.
    ColMajor { align_bytes: usize },
    /// GEMM panel-packed format (mr×kc blocks for B matrix).
    PanelPacked { mr: usize, nr: usize },
    /// Head-split: [seq, heads, head_dim] instead of [seq, hidden].
    HeadSplit { num_heads: usize, head_dim: usize },
    /// Interleaved pairs: (gate[i], up[i]) alternating.
    InterleavedPairs,
    /// Intel AMX tile: 16×32 BF16 format.
    AmxTileBF16 { rows: usize, cols: usize },
    /// GPU shared memory tile with padding for bank-conflict avoidance.
    SharedMemTile { tile_rows: usize, tile_cols: usize, padding_bytes: usize },
    /// GPU TMA 2D-compatible layout (128-byte aligned).
    TmaAligned2D { tile_m: usize, tile_n: usize },
    /// VNNI u8×i4 packed format.
    VnniPacked4,
    /// Any layout acceptable (element-wise ops have no preference).
    Any,
}

impl LayoutConstraint {
    /// Check if two layouts are compatible (one can be consumed by the other
    /// without a data transformation).
    pub fn compatible_with(&self, other: &LayoutConstraint) -> bool {
        match (self, other) {
            // Any is always compatible.
            (LayoutConstraint::Any, _) | (_, LayoutConstraint::Any) => true,
            // Same type → compatible.
            (a, b) if a == b => true,
            // RowMajor with different alignment → compatible (lower alignment is fine).
            (LayoutConstraint::RowMajor { align_bytes: a }, LayoutConstraint::RowMajor { align_bytes: b }) => a <= b,
            // HeadSplit is a reshaped view of RowMajor — compatible for element-wise access.
            (LayoutConstraint::RowMajor { .. }, LayoutConstraint::HeadSplit { .. }) => true,
            (LayoutConstraint::HeadSplit { .. }, LayoutConstraint::RowMajor { .. }) => true,
            // Everything else requires transformation.
            _ => false,
        }
    }
}

/// Operator pattern that an acceleration feature can accelerate.
#[derive(Debug, Clone, PartialEq)]
pub enum ApplicablePattern {
    /// Any GEMM (Gemm, GemmBias, QuantGemm).
    Gemm,
    /// Elementwise operation (Silu, Add, Mul, etc.).
    Elementwise,
    /// Reduction operation (Argmax, Softmax, etc.).
    Reduction,
    /// Normalization (RmsNorm, LayerNorm).
    Norm,
    /// Injective multi-output (RoPE, Reshape).
    Injective,
    /// Attention (MHA, GQA).
    Attention,
    /// Any op — universally applicable.
    Any,
}

/// Benefit function type: takes bottleneck type and returns benefit multiplier.
pub type BenefitFn = fn(BottleneckType) -> f64;

/// A single acceleration feature declaration.
#[derive(Debug, Clone)]
pub struct AccelerationDecl {
    /// Unique identifier.
    pub id: AccelerationId,
    /// Hardware requirement.
    pub hw_req: HardwareRequirement,
    /// Operator patterns this feature can accelerate.
    pub applicable_patterns: Vec<ApplicablePattern>,
    /// Acceptable input layouts (multiple alternatives allowed).
    pub input_layouts: Vec<LayoutConstraint>,
    /// Output layout when using this feature.
    pub output_layout: LayoutConstraint,
    /// Benefit multiplier function.
    pub benefit_fn: BenefitFn,
}

/// The acceleration registry — collection of all known acceleration features.
pub struct AccelerationRegistry {
    entries: Vec<AccelerationDecl>,
}

impl AccelerationRegistry {
    /// Create a registry populated with all built-in acceleration features.
    pub fn new() -> Self {
        let mut reg = AccelerationRegistry { entries: Vec::new() };

        // ── x86_64 AVX-512 GEMM ──
        reg.register(AccelerationDecl {
            id: "avx512_gemm_f32",
            hw_req: HardwareRequirement::Avx512,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![
                LayoutConstraint::RowMajor { align_bytes: 64 },
                LayoutConstraint::PanelPacked { mr: 14, nr: 32 },
            ],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.5,
                BottleneckType::ComputeBound { .. } => 1.0,
                BottleneckType::LatencyBound { .. } => 0.8,
            },
        });

        // ── x86_64 AVX-512 GEMM with HeadSplit output ──
        reg.register(AccelerationDecl {
            id: "avx512_gemm_headsplit",
            hw_req: HardwareRequirement::Avx512,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![
                LayoutConstraint::RowMajor { align_bytes: 64 },
            ],
            output_layout: LayoutConstraint::Any, // Can output any layout via store stride
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.8,
                BottleneckType::ComputeBound { .. } => 1.2,
                BottleneckType::LatencyBound { .. } => 1.0,
            },
        });

        // ── x86_64 AMX tile BF16 GEMM ──
        reg.register(AccelerationDecl {
            id: "amx_tile_bf16",
            hw_req: HardwareRequirement::Avx512Amx,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![
                LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 },
            ],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.5,
                BottleneckType::ComputeBound { .. } => 2.0,
                BottleneckType::LatencyBound { .. } => 1.0,
            },
        });

        // ── x86_64 AVX2 GEMM ──
        reg.register(AccelerationDecl {
            id: "avx2_gemm_f32",
            hw_req: HardwareRequirement::Avx2,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![
                LayoutConstraint::RowMajor { align_bytes: 32 },
                LayoutConstraint::PanelPacked { mr: 6, nr: 8 },
            ],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 32 },
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.3,
                BottleneckType::ComputeBound { .. } => 1.0,
                BottleneckType::LatencyBound { .. } => 0.8,
            },
        });

        // ── ARM NEON GEMM ──
        reg.register(AccelerationDecl {
            id: "neon_gemm_f32",
            hw_req: HardwareRequirement::Neon,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![
                LayoutConstraint::RowMajor { align_bytes: 16 },
            ],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 16 },
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.3,
                BottleneckType::ComputeBound { .. } => 1.0,
                BottleneckType::LatencyBound { .. } => 0.8,
            },
        });

        // ── ARM SVE GEMM ──
        reg.register(AccelerationDecl {
            id: "sve_gemm_f32",
            hw_req: HardwareRequirement::Sve,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![
                LayoutConstraint::RowMajor { align_bytes: 16 },
            ],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 16 },
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.5,
                BottleneckType::ComputeBound { .. } => 1.2,
                BottleneckType::LatencyBound { .. } => 1.0,
            },
        });

        // ── Elementwise (any hardware, no special layout) ──
        reg.register(AccelerationDecl {
            id: "simd_elementwise",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Elementwise],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 1.0,
        });

        // ── Reduction (any hardware) ──
        reg.register(AccelerationDecl {
            id: "simd_reduction",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Reduction],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 1.0,
        });

        // ── Norm (any hardware) ──
        reg.register(AccelerationDecl {
            id: "simd_norm",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Norm],
            input_layouts: vec![LayoutConstraint::RowMajor { align_bytes: 64 }],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            benefit_fn: |bn| match bn {
                BottleneckType::MemoryBound { .. } => 1.2,
                _ => 1.0,
            },
        });

        // ── Injective (any hardware) ──
        reg.register(AccelerationDecl {
            id: "simd_injective",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Injective],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 1.0,
        });

        // ── Attention (any hardware) ──
        reg.register(AccelerationDecl {
            id: "simd_attention",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Attention],
            input_layouts: vec![
                LayoutConstraint::HeadSplit { num_heads: 0, head_dim: 0 },
                LayoutConstraint::RowMajor { align_bytes: 64 },
            ],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            benefit_fn: |_| 2.0,
        });

        reg
    }

    /// Register a new acceleration feature.
    pub fn register(&mut self, decl: AccelerationDecl) {
        self.entries.push(decl);
    }

    /// Query all features applicable to the given OpClass on the given device.
    pub fn query(
        &self,
        op_class: OpClass,
        device: &DeviceProfile,
    ) -> Vec<&AccelerationDecl> {
        let pattern = match op_class {
            OpClass::Gemm => ApplicablePattern::Gemm,
            OpClass::ElemWise => ApplicablePattern::Elementwise,
            OpClass::Reduction => ApplicablePattern::Reduction,
            OpClass::Injective => ApplicablePattern::Injective,
            OpClass::Opaque => ApplicablePattern::Any,
        };

        self.entries.iter()
            .filter(|e| {
                e.hw_req.satisfied_by(device)
                    && (e.applicable_patterns.contains(&pattern)
                        || e.applicable_patterns.contains(&ApplicablePattern::Any))
            })
            .collect()
    }

    /// Get the highest-benefit feature for an OpClass on the given device.
    pub fn best_for(
        &self,
        op_class: OpClass,
        device: &DeviceProfile,
        bottleneck: BottleneckType,
    ) -> Option<&AccelerationDecl> {
        let candidates = self.query(op_class, device);
        candidates.into_iter()
            .max_by(|a, b| {
                let ba = (a.benefit_fn)(bottleneck);
                let bb = (b.benefit_fn)(bottleneck);
                ba.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Number of registered features.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_populated() {
        let reg = AccelerationRegistry::new();
        assert!(reg.len() >= 10, "Registry should have at least 10 entries, got {}", reg.len());
    }

    #[test]
    fn test_query_gemm_avx2() {
        let reg = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let results = reg.query(OpClass::Gemm, &device);
        // Should find at least one GEMM feature (either AVX2 or AVX-512 depending on CPU).
        assert!(!results.is_empty(), "Should find at least one GEMM acceleration feature");
        for r in &results {
            assert!(r.applicable_patterns.contains(&ApplicablePattern::Gemm));
        }
    }

    #[test]
    fn test_query_elementwise_any() {
        let reg = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let results = reg.query(OpClass::ElemWise, &device);
        assert!(!results.is_empty(), "Elementwise should always find simd_elementwise");
    }

    #[test]
    fn test_best_for_gemm() {
        let reg = AccelerationRegistry::new();
        let device = DeviceProfile::detect();
        let best = reg.best_for(OpClass::Gemm, &device, BottleneckType::MemoryBound { bandwidth_utilization: 0.5 });
        assert!(best.is_some(), "Should find a best GEMM feature");
    }

    #[test]
    fn test_layout_compatibility() {
        assert!(LayoutConstraint::Any.compatible_with(&LayoutConstraint::RowMajor { align_bytes: 64 }));
        assert!(LayoutConstraint::RowMajor { align_bytes: 64 }.compatible_with(&LayoutConstraint::Any));
        assert!(LayoutConstraint::RowMajor { align_bytes: 32 }.compatible_with(&LayoutConstraint::RowMajor { align_bytes: 64 }));
        assert!(LayoutConstraint::RowMajor { align_bytes: 64 }.compatible_with(&LayoutConstraint::HeadSplit { num_heads: 32, head_dim: 128 }));
    }

    #[test]
    fn test_hw_req_satisfied() {
        let device = DeviceProfile::detect();
        // Any should always be satisfied.
        assert!(HardwareRequirement::Any.satisfied_by(&device));
    }

    // @trace TEST-AR-07 [req:REQ-CG] [level:unit]
    #[test]
    fn test_layout_any_compatible_with_all_variants() {
        // Arrange
        let any = LayoutConstraint::Any;
        let variants = vec![
            LayoutConstraint::ColMajor { align_bytes: 32 },
            LayoutConstraint::PanelPacked { mr: 6, nr: 8 },
            LayoutConstraint::InterleavedPairs,
            LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 },
            LayoutConstraint::SharedMemTile { tile_rows: 16, tile_cols: 16, padding_bytes: 4 },
            LayoutConstraint::TmaAligned2D { tile_m: 128, tile_n: 128 },
            LayoutConstraint::VnniPacked4,
        ];

        // Act & Assert — Any is compatible with every layout variant, both directions.
        for variant in &variants {
            assert!(any.compatible_with(variant),
                "Any should be compatible with {:?}", variant);
            assert!(variant.compatible_with(&any),
                "{:?} should be compatible with Any", variant);
        }
    }

    // @trace TEST-AR-08 [req:REQ-CG] [level:unit]
    #[test]
    fn test_layout_row_major_higher_align_not_compatible_with_lower() {
        // Arrange
        let high_align = LayoutConstraint::RowMajor { align_bytes: 128 };
        let low_align = LayoutConstraint::RowMajor { align_bytes: 32 };

        // Act & Assert — only the lower-align → higher-align direction is compatible.
        assert!(low_align.compatible_with(&high_align),
            "Lower alignment (32) should be compatible with higher alignment (128)");
        assert!(!high_align.compatible_with(&low_align),
            "Higher alignment (128) should NOT be compatible with lower alignment (32)");
    }

    // @trace TEST-AR-09 [req:REQ-CG] [level:unit]
    #[test]
    fn test_layout_incompatible_pairs() {
        // Arrange
        let row_major = LayoutConstraint::RowMajor { align_bytes: 64 };
        let col_major = LayoutConstraint::ColMajor { align_bytes: 64 };
        let panel = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };
        let interleaved = LayoutConstraint::InterleavedPairs;
        let amx = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };
        let vnni = LayoutConstraint::VnniPacked4;

        // Act & Assert — all these pairs require transformation.
        assert!(!row_major.compatible_with(&col_major),
            "RowMajor and ColMajor should be incompatible");
        assert!(!row_major.compatible_with(&panel),
            "RowMajor and PanelPacked should be incompatible");
        assert!(!row_major.compatible_with(&interleaved),
            "RowMajor and InterleavedPairs should be incompatible");
        assert!(!row_major.compatible_with(&amx),
            "RowMajor and AmxTileBF16 should be incompatible");
        assert!(!row_major.compatible_with(&vnni),
            "RowMajor and VnniPacked4 should be incompatible");
        assert!(!col_major.compatible_with(&panel),
            "ColMajor and PanelPacked should be incompatible");
    }

    // @trace TEST-AR-10 [req:REQ-CG] [level:unit]
    #[test]
    fn test_layout_headsplit_rowmajor_bidirectional() {
        // Arrange
        let row = LayoutConstraint::RowMajor { align_bytes: 64 };
        let head = LayoutConstraint::HeadSplit { num_heads: 32, head_dim: 128 };

        // Act & Assert — HeadSplit and RowMajor are compatible in both directions.
        assert!(row.compatible_with(&head),
            "RowMajor should be compatible with HeadSplit");
        assert!(head.compatible_with(&row),
            "HeadSplit should be compatible with RowMajor");
    }

    // @trace TEST-AR-11 [req:REQ-CG] [level:unit]
    #[test]
    fn test_gpu_hw_requirements_never_satisfied_on_cpu() {
        // Arrange
        let device = DeviceProfile::detect();
        let gpu_reqs = vec![
            HardwareRequirement::GpuSm80,
            HardwareRequirement::GpuSm90,
            HardwareRequirement::GpuSm100,
        ];

        // Act & Assert — GPU requirements are never satisfied on CPU-only builds.
        for req in &gpu_reqs {
            assert!(!req.satisfied_by(&device),
                "{:?} should not be satisfied on a CPU-only build", req);
        }
    }

    // @trace TEST-AR-12 [req:REQ-CG] [level:unit]
    #[test]
    fn test_query_reduction_injective_and_opaque() {
        // Arrange
        let reg = AccelerationRegistry::new();
        let device = DeviceProfile::detect();

        // Act
        let reduction = reg.query(OpClass::Reduction, &device);
        let injective = reg.query(OpClass::Injective, &device);
        let opaque = reg.query(OpClass::Opaque, &device);

        // Assert — Reduction and Injective have dedicated "Any"-hw entries.
        assert!(!reduction.is_empty(),
            "Reduction should always find simd_reduction");
        assert!(!injective.is_empty(),
            "Injective should always find simd_injective");
        // Opaque maps to ApplicablePattern::Any; no built-in feature declares that pattern,
        // so the result is legitimately empty on a default registry.
        assert!(opaque.is_empty(),
            "Opaque has no matching ApplicablePattern::Any feature in default registry");
    }

    // @trace TEST-AR-13 [req:REQ-CG] [level:unit]
    #[test]
    fn test_best_for_picks_highest_benefit() {
        // Arrange — register two features for the same pattern with known benefit deltas.
        let mut reg = AccelerationRegistry { entries: Vec::new() };
        reg.register(AccelerationDecl {
            id: "low_benefit_gemm",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 1.0,
        });
        reg.register(AccelerationDecl {
            id: "high_benefit_gemm",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 5.0,
        });
        let device = DeviceProfile::detect();

        // Act
        let best = reg.best_for(OpClass::Gemm, &device, BottleneckType::MemoryBound { bandwidth_utilization: 0.5 });

        // Assert
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "high_benefit_gemm",
            "best_for should pick the feature with the highest benefit multiplier");
    }

    // @trace TEST-AR-14 [req:REQ-CG] [level:unit]
    #[test]
    fn test_best_for_compute_bound_and_latency_bound() {
        // Arrange
        let reg = AccelerationRegistry::new();
        let device = DeviceProfile::detect();

        // Act
        let compute_best = reg.best_for(OpClass::Gemm, &device,
            BottleneckType::ComputeBound { compute_utilization: 0.9 });
        let latency_best = reg.best_for(OpClass::Gemm, &device,
            BottleneckType::LatencyBound { estimated_latency_ns: 100.0 });

        // Assert — both should find a feature (at minimum the Any-hw GEMM features).
        assert!(compute_best.is_some(),
            "Should find a GEMM feature for compute-bound bottleneck");
        assert!(latency_best.is_some(),
            "Should find a GEMM feature for latency-bound bottleneck");
    }

    // @trace TEST-AR-15 [req:REQ-CG] [level:unit]
    #[test]
    fn test_empty_registry() {
        // Arrange
        let reg = AccelerationRegistry { entries: Vec::new() };

        // Act & Assert
        assert!(reg.is_empty(), "New empty registry should be empty");
        assert_eq!(reg.len(), 0, "Empty registry length should be 0");

        let device = DeviceProfile::detect();
        let results = reg.query(OpClass::Gemm, &device);
        assert!(results.is_empty(), "Query on empty registry should return no results");

        let best = reg.best_for(OpClass::Gemm, &device,
            BottleneckType::MemoryBound { bandwidth_utilization: 0.5 });
        assert!(best.is_none(), "best_for on empty registry should return None");
    }

    // @trace TEST-AR-16 [req:REQ-CG] [level:unit]
    #[test]
    fn test_register_increments_len() {
        // Arrange
        let mut reg = AccelerationRegistry { entries: Vec::new() };
        let initial_len = reg.len();

        let decl = AccelerationDecl {
            id: "test_custom",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 3.0,
        };

        // Act
        reg.register(decl);

        // Assert
        assert_eq!(reg.len(), initial_len + 1);
        assert!(!reg.is_empty());
    }

    // @trace TEST-AR-17 [req:REQ-CG] [level:unit]
    #[test]
    fn test_query_filters_by_hw_requirement() {
        // Arrange — register one GPU-only feature.
        let mut reg = AccelerationRegistry { entries: Vec::new() };
        reg.register(AccelerationDecl {
            id: "gpu_only_gemm",
            hw_req: HardwareRequirement::GpuSm80,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 10.0,
        });
        let device = DeviceProfile::detect();

        // Act
        let results = reg.query(OpClass::Gemm, &device);

        // Assert — GPU feature should not appear on CPU-only build.
        assert!(results.is_empty(),
            "GPU-only feature should not match on CPU device profile");
    }

    // @trace TEST-AR-18 [req:REQ-CG] [level:unit]
    #[test]
    fn test_query_applicable_pattern_any_matches_all_classes() {
        // Arrange — register a feature with ApplicablePattern::Any.
        let mut reg = AccelerationRegistry { entries: Vec::new() };
        reg.register(AccelerationDecl {
            id: "universal_accel",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Any],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 1.0,
        });
        let device = DeviceProfile::detect();

        // Act & Assert — it should be returned for every OpClass.
        for op_class in &[OpClass::Gemm, OpClass::ElemWise, OpClass::Reduction, OpClass::Injective, OpClass::Opaque] {
            let results = reg.query(*op_class, &device);
            assert!(results.iter().any(|r| r.id == "universal_accel"),
                "ApplicablePattern::Any should match OpClass {:?}", op_class);
        }
    }

    // @trace TEST-AR-19 [req:REQ-CG] [level:unit]
    #[test]
    fn test_benefit_fn_varies_by_bottleneck_type() {
        // Arrange
        let reg = AccelerationRegistry::new();
        let device = DeviceProfile::detect();

        let mem = BottleneckType::MemoryBound { bandwidth_utilization: 0.5 };
        let compute = BottleneckType::ComputeBound { compute_utilization: 0.9 };
        let latency = BottleneckType::LatencyBound { estimated_latency_ns: 100.0 };

        // Act — find the avx512_gemm_f32 benefit values (it has distinct values per bottleneck).
        let entry = reg.entries.iter().find(|e| e.id == "avx512_gemm_f32");

        // Assert — skip if not available on this hardware; otherwise verify differentiation.
        if let Some(e) = entry {
            let b_mem = (e.benefit_fn)(mem);
            let b_compute = (e.benefit_fn)(compute);
            let b_latency = (e.benefit_fn)(latency);
            // MemoryBound has highest benefit for this entry.
            assert!(b_mem > b_compute,
                "avx512_gemm_f32: memory-bound benefit ({}) should exceed compute-bound ({})",
                b_mem, b_compute);
            assert!(b_compute > b_latency,
                "avx512_gemm_f32: compute-bound benefit ({}) should exceed latency-bound ({})",
                b_compute, b_latency);
        }
    }

    // @trace TEST-AR-20 [req:REQ-CG] [level:unit]
    #[test]
    fn test_hw_requirement_avx2_satisfied_by_avx512() {
        // Arrange — AVX-512 CPUs satisfy the AVX2 requirement (backward compatible).
        let device = DeviceProfile::detect();
        let avx2_req = HardwareRequirement::Avx2;

        // Act
        let satisfied = avx2_req.satisfied_by(&device);

        // Assert — if the device has AVX-512 or AVX2, the Avx2 requirement should be met.
        if matches!(device.isa, IsaLevel::Avx2 | IsaLevel::Avx512 | IsaLevel::Avx512Amx) {
            assert!(satisfied,
                "Avx2 requirement should be satisfied on {:?} device", device.isa);
        }
    }

    // @trace TEST-AR-21 [req:REQ-CG] [level:unit]
    #[test]
    fn test_hw_requirement_neon_satisfied_by_sve() {
        // Arrange — SVE CPUs satisfy the NEON requirement (backward compatible).
        let device = DeviceProfile::detect();
        let neon_req = HardwareRequirement::Neon;

        // Act
        let satisfied = neon_req.satisfied_by(&device);

        // Assert — if the device has NEON, SVE, SVE2, or NeonAmx, NEON req is met.
        if matches!(device.isa, IsaLevel::Neon | IsaLevel::Sve | IsaLevel::Sve2 | IsaLevel::NeonAmx) {
            assert!(satisfied,
                "Neon requirement should be satisfied on {:?} device", device.isa);
        }
    }

    // @trace TEST-AR-22 [req:REQ-CG] [level:unit]
    #[test]
    fn test_hw_requirement_avx512_not_satisfied_by_avx2() {
        // Arrange
        let device = DeviceProfile::detect();
        let avx512_req = HardwareRequirement::Avx512;

        // Act
        let satisfied = avx512_req.satisfied_by(&device);

        // Assert — Avx512 requirement should NOT be met on an Avx2-only device.
        if matches!(device.isa, IsaLevel::Avx2) {
            assert!(!satisfied,
                "Avx512 requirement should NOT be satisfied on Avx2-only device");
        }
    }

    // @trace TEST-AR-23 [req:REQ-CG] [level:unit]
    #[test]
    fn test_hw_requirement_sve_not_satisfied_by_neon() {
        // Arrange
        let device = DeviceProfile::detect();
        let sve_req = HardwareRequirement::Sve;

        // Act
        let satisfied = sve_req.satisfied_by(&device);

        // Assert — SVE requirement should NOT be met on a NEON-only device.
        if matches!(device.isa, IsaLevel::Neon) {
            assert!(!satisfied,
                "Sve requirement should NOT be satisfied on Neon-only device");
        }
    }

    // @trace TEST-AR-24 [req:REQ-CG] [level:unit]
    #[test]
    fn test_acceleration_decl_debug_output() {
        // Arrange
        let decl = AccelerationDecl {
            id: "test_debug_decl",
            hw_req: HardwareRequirement::Avx512,
            applicable_patterns: vec![ApplicablePattern::Gemm, ApplicablePattern::Norm],
            input_layouts: vec![LayoutConstraint::RowMajor { align_bytes: 64 }],
            output_layout: LayoutConstraint::RowMajor { align_bytes: 64 },
            benefit_fn: |_| 2.5,
        };

        // Act
        let debug_str = format!("{:?}", decl);

        // Assert — Debug output should contain the id and key fields.
        assert!(debug_str.contains("test_debug_decl"),
            "Debug output should contain the acceleration id");
        assert!(debug_str.contains("Avx512"),
            "Debug output should contain the hardware requirement");
        assert!(debug_str.contains("Gemm"),
            "Debug output should contain the applicable patterns");
    }

    // @trace TEST-AR-25 [req:REQ-CG] [level:unit]
    #[test]
    fn test_best_for_empty_registry_returns_none() {
        // Arrange
        let reg = AccelerationRegistry { entries: Vec::new() };
        let device = DeviceProfile::detect();

        // Act
        let result = reg.best_for(
            OpClass::ElemWise,
            &device,
            BottleneckType::ComputeBound { compute_utilization: 0.5 },
        );

        // Assert
        assert!(result.is_none(),
            "best_for on empty registry should return None for any OpClass");
    }

    // @trace TEST-AR-26 [req:REQ-CG] [level:unit]
    #[test]
    fn test_query_returns_only_matching_patterns() {
        // Arrange — register features for Gemm only and Elementwise only.
        let mut reg = AccelerationRegistry { entries: Vec::new() };
        reg.register(AccelerationDecl {
            id: "gemm_only",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 2.0,
        });
        reg.register(AccelerationDecl {
            id: "elem_only",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Elementwise],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 1.5,
        });
        let device = DeviceProfile::detect();

        // Act — query for Injective, which neither feature supports.
        let injective_results = reg.query(OpClass::Injective, &device);

        // Assert — no features should match Injective.
        assert!(injective_results.is_empty(),
            "Query for Injective should return empty when no Injective features are registered");

        // Act — query for Gemm.
        let gemm_results = reg.query(OpClass::Gemm, &device);

        // Assert — only the Gemm feature should match.
        assert_eq!(gemm_results.len(), 1,
            "Query for Gemm should return exactly one feature");
        assert_eq!(gemm_results[0].id, "gemm_only");
    }

    // @trace TEST-AR-27 [req:REQ-CG] [level:unit]
    #[test]
    fn test_layout_same_type_compatible() {
        // Arrange — identical layout constraints.
        let a = LayoutConstraint::ColMajor { align_bytes: 64 };
        let b = LayoutConstraint::ColMajor { align_bytes: 64 };
        let panel_a = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };
        let panel_b = LayoutConstraint::PanelPacked { mr: 14, nr: 32 };
        let amx_a = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };
        let amx_b = LayoutConstraint::AmxTileBF16 { rows: 16, cols: 32 };

        // Act & Assert — same types should be compatible.
        assert!(a.compatible_with(&b), "Identical ColMajor should be compatible");
        assert!(panel_a.compatible_with(&panel_b), "Identical PanelPacked should be compatible");
        assert!(amx_a.compatible_with(&amx_b), "Identical AmxTileBF16 should be compatible");
    }

    // @trace TEST-AR-28 [req:REQ-CG] [level:unit]
    #[test]
    fn test_layout_different_col_major_align_incompatible() {
        // Arrange — ColMajor with different alignments is NOT compatible
        // (unlike RowMajor, ColMajor does not have the alignment relaxation rule).
        let low = LayoutConstraint::ColMajor { align_bytes: 32 };
        let high = LayoutConstraint::ColMajor { align_bytes: 64 };

        // Act & Assert — different alignment on ColMajor requires transformation.
        assert!(!low.compatible_with(&high),
            "ColMajor with different alignment should be incompatible");
        assert!(!high.compatible_with(&low),
            "ColMajor with different alignment should be incompatible (reverse)");
    }

    // @trace TEST-AR-29 [req:REQ-CG] [level:unit]
    #[test]
    fn test_best_for_ties_prefers_any_result() {
        // Arrange — two features with identical benefit, both Any-hw + Gemm.
        let mut reg = AccelerationRegistry { entries: Vec::new() };
        reg.register(AccelerationDecl {
            id: "tie_a",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 3.0,
        });
        reg.register(AccelerationDecl {
            id: "tie_b",
            hw_req: HardwareRequirement::Any,
            applicable_patterns: vec![ApplicablePattern::Gemm],
            input_layouts: vec![LayoutConstraint::Any],
            output_layout: LayoutConstraint::Any,
            benefit_fn: |_| 3.0,
        });
        let device = DeviceProfile::detect();

        // Act
        let best = reg.best_for(OpClass::Gemm, &device,
            BottleneckType::MemoryBound { bandwidth_utilization: 0.5 });

        // Assert — a result should be returned (either one); both have equal benefit.
        assert!(best.is_some(),
            "best_for should return a result even when benefits are tied");
        let id = best.unwrap().id;
        assert!(id == "tie_a" || id == "tie_b",
            "best_for should return one of the tied features, got {}", id);
    }
}
