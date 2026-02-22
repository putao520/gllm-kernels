//! Execution Planner — transforms LayerIR + DeviceProfile into an ExecutionPlan.
//!
//! The planner decides all performance-critical parameters at compile time:
//! GEMM blocking, fusion strategy, tiling, thread count, prefetch distances.
//! At runtime, the compiled layer executes with zero decisions.

use std::collections::HashMap;
use crate::compiler::ir::{LayerIR, LayerArch};
use crate::dispatch::device_profile::DeviceProfile;
use crate::traits::Activation;

/// A GEMM shape key for blocking parameter lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Microkernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrokernelChoice {
    pub mr: usize,
    pub nr: usize,
}

/// Fusion decisions — which operator pairs are fused in the compiled layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionDecision {
    /// RMSNorm output feeds directly into GEMM without memory writeback
    RmsNormIntoGemm,
    /// GEMM epilogue fuses bias + activation
    GemmBiasAct(Activation),
    /// QKV three GEMMs share input, single pack_a
    QkvSharedInput,
    /// Attention score + softmax + V matmul fused (FlashAttention-style tiling)
    FlashAttention,
    /// Gate and Up GEMMs fused with SiLU: SiLU(gate) * up
    SwiGluFusion,
    /// Gate and Up GEMMs fused with GELU: GELU(gate) * up (Gemma GeGLU)
    GeGluFusion,
}

/// Complete execution plan for a compiled transformer layer.
///
/// Every performance parameter is determined here. The codegen phase
/// translates this plan into machine code with all values baked as immediates.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Hardware profile used for planning
    pub profile: DeviceProfile,

    // ── GEMM blocking (per shape) ──
    /// (KC, MC, NC) for each distinct GEMM shape in the layer
    pub gemm_blocking: HashMap<GemmShape, (usize, usize, usize)>,

    // ── Threading ──
    /// Thread count for compute-bound regions (GEMM)
    pub num_threads: usize,

    // ── Microkernel ──
    pub microkernel: MicrokernelChoice,

    // ── Prefetch ──
    pub prefetch_a_l1: usize,
    pub prefetch_b_l1: usize,
    pub prefetch_a_l2: usize,

    // ── Unrolling ──
    pub k_unroll: usize,

    // ── Attention tiling ──
    pub attn_tile_q: usize,
    pub attn_tile_kv: usize,

    // ── Buffer layout ──
    /// Total scratchpad bytes needed by the compiled layer
    pub scratchpad_bytes: usize,

    // ── Fusion decisions ──
    pub fusions: Vec<FusionDecision>,
}

impl ExecutionPlan {
    /// Build an execution plan from a LayerIR and DeviceProfile.
    pub fn build(ir: &LayerIR, profile: &DeviceProfile) -> Self {
        let kc = &profile.kernel_config;
        let (mr, nr) = (kc.mr, kc.nr);

        // Collect all GEMM shapes in this layer
        let mut gemm_blocking = HashMap::new();
        let gemm_shapes = collect_gemm_shapes(ir);
        for shape in &gemm_shapes {
            let blocking = compute_blocking(shape, kc.kc, kc.mc, kc.nc, mr, nr);
            gemm_blocking.insert(*shape, blocking);
        }

        // Threading: use physical cores for compute-bound
        let num_threads = profile.physical_cores;

        // Prefetch from kernel config
        let prefetch_a_l1 = kc.pf_distance_a;
        let prefetch_b_l1 = kc.pf_distance_b;
        let prefetch_a_l2 = kc.pf_distance_a * 2;

        // K unroll: 4 for AVX2, 8 for AVX-512
        let k_unroll = if kc.use_avx512 { 8 } else { 4 };

        // Attention tiling
        let attn_tile_q = 32.min(ir.max_seq);
        let attn_tile_kv = 256.min(ir.max_seq);

        // Scratchpad: sum of all intermediate buffers
        let scratchpad_bytes = compute_scratchpad(ir);

        // Fusion decisions
        let fusions = plan_fusions(ir);

        ExecutionPlan {
            profile: profile.clone(),
            gemm_blocking,
            num_threads,
            microkernel: MicrokernelChoice { mr, nr },
            prefetch_a_l1,
            prefetch_b_l1,
            prefetch_a_l2,
            k_unroll,
            attn_tile_q,
            attn_tile_kv,
            scratchpad_bytes,
            fusions,
        }
    }
}

/// Collect all distinct GEMM shapes in a layer.
fn collect_gemm_shapes(ir: &LayerIR) -> Vec<GemmShape> {
    let h = ir.hidden;
    let q = ir.q_dim();
    let kv = ir.kv_dim();
    let inter = ir.intermediate;

    let mut shapes = vec![
        // QKV projections (M=1 for single token, but plan for max_batch)
        GemmShape { m: ir.max_batch, n: q, k: h },     // Q
        GemmShape { m: ir.max_batch, n: kv, k: h },    // K, V
        // Output projection
        GemmShape { m: ir.max_batch, n: h, k: q },     // O
    ];

    match &ir.arch {
        LayerArch::Decoder | LayerArch::DecoderMoE { .. } => {
            // FFN: gate, up, down
            shapes.push(GemmShape { m: ir.max_batch, n: inter, k: h });  // gate, up
            shapes.push(GemmShape { m: ir.max_batch, n: h, k: inter }); // down
        }
        LayerArch::Encoder => {
            shapes.push(GemmShape { m: ir.max_batch, n: inter, k: h });
            shapes.push(GemmShape { m: ir.max_batch, n: h, k: inter });
        }
    }

    shapes.sort_by_key(|s| (s.m, s.n, s.k));
    shapes.dedup();
    shapes
}

/// Compute BLIS-style blocking for a GEMM shape.
fn compute_blocking(
    shape: &GemmShape,
    default_kc: usize,
    default_mc: usize,
    default_nc: usize,
    mr: usize,
    nr: usize,
) -> (usize, usize, usize) {
    let kc = default_kc.min(shape.k);
    let mc = default_mc.min(shape.m);
    let nc = default_nc.min(shape.n);

    // Align to microkernel tile
    let mc = (mc / mr) * mr;
    let nc = (nc / nr) * nr;

    (kc.max(1), mc.max(mr), nc.max(nr))
}

/// Compute total scratchpad bytes for a layer.
fn compute_scratchpad(ir: &LayerIR) -> usize {
    let elem = ir.dtype.size_bytes();
    let h = ir.hidden;
    let q = ir.q_dim();
    let kv = ir.kv_dim();
    let inter = ir.intermediate;
    let b = ir.max_batch;

    // Normed input
    let normed = b * h * elem;
    // QKV outputs
    let qkv = b * (q + 2 * kv) * elem;
    // Attention output
    let attn = b * q * elem;
    // FFN intermediates (gate + up)
    let ffn = b * 2 * inter * elem;
    // Attention scores (per head)
    let scores = b * ir.num_heads * ir.max_seq * elem;

    // We reuse buffers, so take the max of non-overlapping phases
    let phase_attn = normed + qkv + attn + scores;
    let phase_ffn = normed + ffn;

    phase_attn.max(phase_ffn)
}

/// Determine fusion opportunities for a layer.
fn plan_fusions(ir: &LayerIR) -> Vec<FusionDecision> {
    let mut fusions = Vec::new();

    // QKV shared input is always beneficial
    fusions.push(FusionDecision::QkvSharedInput);

    match &ir.arch {
        LayerArch::Decoder | LayerArch::DecoderMoE { .. } => {
            // Gated FFN fusion: SwiGLU or GeGLU depending on activation
            match ir.activation {
                Activation::GeGlu => fusions.push(FusionDecision::GeGluFusion),
                _ => fusions.push(FusionDecision::SwiGluFusion),
            }
        }
        LayerArch::Encoder => {
            // GELU fusion for encoder FFN
            fusions.push(FusionDecision::GemmBiasAct(Activation::Gelu));
        }
    }

    // FlashAttention tiling for long sequences
    if ir.max_seq > 128 {
        fusions.push(FusionDecision::FlashAttention);
    }

    // RMSNorm→GEMM fusion (always beneficial, avoids one memory round-trip)
    fusions.push(FusionDecision::RmsNormIntoGemm);

    fusions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::LayerIR;
    use crate::inference::types::ModelConfig;
    use crate::dispatch::DeviceProfile;

    #[test]
    fn test_execution_plan_build() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile);

        assert!(plan.num_threads >= 1);
        assert!(plan.scratchpad_bytes > 0);
        assert!(!plan.fusions.is_empty());
        assert!(!plan.gemm_blocking.is_empty());
        assert!(plan.microkernel.mr >= 4);
        assert!(plan.microkernel.nr >= 4);

        eprintln!("Plan: {} threads, {} scratchpad bytes, {} fusions, {} GEMM shapes",
            plan.num_threads,
            plan.scratchpad_bytes,
            plan.fusions.len(),
            plan.gemm_blocking.len(),
        );
    }

    #[test]
    fn test_fusions_decoder() {
        let config = ModelConfig::llama_7b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile);

        assert!(plan.fusions.contains(&FusionDecision::QkvSharedInput));
        assert!(plan.fusions.contains(&FusionDecision::SwiGluFusion));
        assert!(plan.fusions.contains(&FusionDecision::RmsNormIntoGemm));
        assert!(plan.fusions.contains(&FusionDecision::FlashAttention));
    }

    #[test]
    fn test_fusions_gemma_geglu() {
        let config = ModelConfig::gemma_2b();
        let ir = LayerIR::from_model_config(&config, 1);
        let profile = DeviceProfile::detect();
        let plan = ExecutionPlan::build(&ir, &profile);

        assert!(plan.fusions.contains(&FusionDecision::GeGluFusion));
        assert!(!plan.fusions.contains(&FusionDecision::SwiGluFusion));
    }
}
