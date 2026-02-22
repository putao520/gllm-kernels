//! Operation semantics — classifies ops for fusion and scheduling decisions.
//!
//! Each `OpKind` is classified by:
//! - `OpSemantics`: elementwise / gemm / reduction / opaque
//! - `BottleneckType`: compute-bound or memory-bound
//! - Arithmetic intensity (FLOP/byte) for roofline analysis
//!
//! The fusion pass (Phase 2) uses these classifications to decide which
//! adjacent ops can be fused without violating data dependencies.

use crate::compiler::graph::OpKind;

/// Semantic classification of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpSemantics {
    /// Element-independent: out[i] = f(in[i], ...).
    /// Can always be fused into a preceding op's epilogue.
    Elementwise,
    /// Matrix multiply: high arithmetic intensity, compute-bound.
    /// Fusion target: can absorb elementwise epilogues (bias, activation).
    Gemm,
    /// Reduction across a dimension (softmax, norm).
    /// Requires full input before producing output — fusion barrier.
    Reduction,
    /// Opaque / multi-pass (attention, RoPE).
    /// Cannot be fused with neighbors in general.
    Opaque,
}

/// Performance bottleneck classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    /// Limited by FMA throughput (GEMM with large K).
    ComputeBound,
    /// Limited by memory bandwidth (elementwise, small GEMM).
    MemoryBound,
}

/// Classify an `OpKind` into its semantic category.
pub fn classify(kind: &OpKind) -> OpSemantics {
    match kind {
        // Elementwise ops
        OpKind::Silu | OpKind::Gelu | OpKind::Add | OpKind::Mul | OpKind::Residual => {
            OpSemantics::Elementwise
        }
        // Gated activations: elementwise (two inputs, one output, no reduction)
        OpKind::SwiGlu | OpKind::GeGlu => OpSemantics::Elementwise,

        // GEMM variants
        OpKind::Gemm { .. } | OpKind::GemmBias { .. } => OpSemantics::Gemm,

        // Reductions (need full input before output)
        OpKind::Softmax | OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => {
            OpSemantics::Reduction
        }

        // Opaque (complex multi-step)
        OpKind::RoPE { .. } | OpKind::Transpose { .. } | OpKind::Reshape { .. } => {
            OpSemantics::Opaque
        }
    }
}

/// Determine the performance bottleneck for an operation.
///
/// Uses the roofline model: if arithmetic intensity > ridge point,
/// the op is compute-bound; otherwise memory-bound.
pub fn bottleneck(kind: &OpKind) -> BottleneckType {
    match kind {
        // GEMM: compute-bound when K is large enough (typical for transformer layers)
        OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => {
            // Arithmetic intensity = 2*M*N*K / (4*(M*K + K*N + M*N))
            // For typical transformer shapes (K≥128), this exceeds any ridge point.
            let flops = 2 * m * n * k;
            let bytes = 4 * (m * k + k * n + m * n);
            if bytes > 0 && (flops / bytes) >= 4 {
                BottleneckType::ComputeBound
            } else {
                BottleneckType::MemoryBound
            }
        }
        // Everything else is memory-bound (bandwidth-limited)
        _ => BottleneckType::MemoryBound,
    }
}

/// Estimate arithmetic intensity (FLOP / byte) for an operation.
///
/// Used by the roofline model to predict whether an op benefits from
/// fusion (memory-bound ops benefit most from eliminating round-trips).
pub fn arithmetic_intensity(kind: &OpKind) -> f64 {
    match kind {
        OpKind::Gemm { m, n, k } | OpKind::GemmBias { m, n, k } => {
            let flops = (2 * m * n * k) as f64;
            let bytes = (4 * (m * k + k * n + m * n)) as f64;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        // Elementwise: 1 FLOP per element, 8 bytes (read + write)
        OpKind::Silu | OpKind::Gelu => {
            // ~10 FLOPs per element (exp approximation), 8 bytes
            10.0 / 8.0
        }
        OpKind::Add | OpKind::Mul | OpKind::Residual => {
            // 1 FLOP, 12 bytes (2 reads + 1 write)
            1.0 / 12.0
        }
        OpKind::SwiGlu | OpKind::GeGlu => {
            // ~12 FLOPs (silu/gelu + mul), 12 bytes (2 reads + 1 write)
            12.0 / 12.0
        }
        OpKind::Softmax => {
            // 3-pass: ~5N FLOPs, ~12N bytes
            5.0 / 12.0
        }
        OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } => {
            // 2-pass: ~5N FLOPs, ~12N bytes
            5.0 / 12.0
        }
        OpKind::RoPE { .. } => {
            // 6 FLOPs per pair, 16 bytes (2 reads + 2 writes + cos/sin)
            6.0 / 16.0
        }
        OpKind::Transpose { .. } | OpKind::Reshape { .. } => {
            // Pure data movement, 0 compute
            0.0
        }
    }
}

/// Whether an elementwise op can be fused as a GEMM epilogue.
///
/// The GEMM microkernel can absorb certain elementwise ops into its
/// store phase (bias add, activation) without extra memory traffic.
pub fn fusable_as_gemm_epilogue(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Add | OpKind::Silu | OpKind::Gelu | OpKind::Residual
    )
}

/// Whether two adjacent ops can potentially be fused.
///
/// Basic fusability rules:
/// - Elementwise after GEMM → GEMM epilogue fusion
/// - Elementwise after Elementwise → loop fusion
/// - Reduction is a fusion barrier (needs full input)
/// - GEMM after GEMM → no fusion (separate tiling)
pub fn can_fuse(producer: &OpKind, consumer: &OpKind) -> bool {
    let prod_sem = classify(producer);
    let cons_sem = classify(consumer);

    match (prod_sem, cons_sem) {
        // GEMM → Elementwise: fuse as epilogue
        (OpSemantics::Gemm, OpSemantics::Elementwise) => true,
        // Elementwise → Elementwise: loop fusion
        (OpSemantics::Elementwise, OpSemantics::Elementwise) => true,
        // Reduction → Elementwise: the reduction output feeds elementwise
        // This is valid (e.g., RmsNorm → scale by weight is already fused)
        (OpSemantics::Reduction, OpSemantics::Elementwise) => true,
        // Everything else: no fusion
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_elementwise() {
        assert_eq!(classify(&OpKind::Silu), OpSemantics::Elementwise);
        assert_eq!(classify(&OpKind::Add), OpSemantics::Elementwise);
        assert_eq!(classify(&OpKind::SwiGlu), OpSemantics::Elementwise);
        assert_eq!(classify(&OpKind::Residual), OpSemantics::Elementwise);
    }

    #[test]
    fn test_classify_gemm() {
        assert_eq!(
            classify(&OpKind::Gemm { m: 1, n: 4096, k: 4096 }),
            OpSemantics::Gemm
        );
        assert_eq!(
            classify(&OpKind::GemmBias { m: 1, n: 4096, k: 4096 }),
            OpSemantics::Gemm
        );
    }

    #[test]
    fn test_classify_reduction() {
        assert_eq!(classify(&OpKind::Softmax), OpSemantics::Reduction);
        assert_eq!(classify(&OpKind::RmsNorm { eps: 1e-5 }), OpSemantics::Reduction);
    }

    #[test]
    fn test_bottleneck_large_gemm() {
        let kind = OpKind::Gemm { m: 128, n: 4096, k: 4096 };
        assert_eq!(bottleneck(&kind), BottleneckType::ComputeBound);
    }

    #[test]
    fn test_bottleneck_small_gemm() {
        // M=1 GEMV: memory-bound
        let kind = OpKind::Gemm { m: 1, n: 4096, k: 4096 };
        assert_eq!(bottleneck(&kind), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_bottleneck_elementwise() {
        assert_eq!(bottleneck(&OpKind::Silu), BottleneckType::MemoryBound);
        assert_eq!(bottleneck(&OpKind::Add), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_arithmetic_intensity_gemm() {
        let ai = arithmetic_intensity(&OpKind::Gemm { m: 128, n: 4096, k: 4096 });
        // Should be high (compute-bound)
        assert!(ai > 10.0, "GEMM AI={ai}, expected >10");
    }

    #[test]
    fn test_arithmetic_intensity_add() {
        let ai = arithmetic_intensity(&OpKind::Add);
        assert!(ai < 1.0, "Add AI={ai}, expected <1");
    }

    #[test]
    fn test_can_fuse_gemm_silu() {
        let gemm = OpKind::Gemm { m: 1, n: 4096, k: 4096 };
        assert!(can_fuse(&gemm, &OpKind::Silu));
        assert!(can_fuse(&gemm, &OpKind::Add));
    }

    #[test]
    fn test_cannot_fuse_gemm_gemm() {
        let g1 = OpKind::Gemm { m: 1, n: 4096, k: 4096 };
        let g2 = OpKind::Gemm { m: 1, n: 4096, k: 4096 };
        assert!(!can_fuse(&g1, &g2));
    }

    #[test]
    fn test_can_fuse_elementwise_chain() {
        assert!(can_fuse(&OpKind::Silu, &OpKind::Mul));
        assert!(can_fuse(&OpKind::Add, &OpKind::Silu));
    }

    #[test]
    fn test_cannot_fuse_into_reduction() {
        assert!(!can_fuse(&OpKind::Add, &OpKind::Softmax));
        assert!(!can_fuse(&OpKind::Silu, &OpKind::RmsNorm { eps: 1e-5 }));
    }

    #[test]
    fn test_fusable_as_gemm_epilogue() {
        assert!(fusable_as_gemm_epilogue(&OpKind::Add));
        assert!(fusable_as_gemm_epilogue(&OpKind::Silu));
        assert!(!fusable_as_gemm_epilogue(&OpKind::SwiGlu));
        assert!(!fusable_as_gemm_epilogue(&OpKind::Softmax));
    }
}
