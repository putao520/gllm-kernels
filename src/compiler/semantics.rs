//! Operation semantics — classifies ops for fusion and scheduling decisions.
//!
//! Each `OpKind` is classified by:
//! - `OpSemantics`: elementwise / gemm / reduction / opaque
//! - `BottleneckType`: compute-bound or memory-bound
//! - Arithmetic intensity (FLOP/byte) for roofline analysis
//!
//! `classify()` is used by `fusion::helpers` to determine elementwise
//! consumers eligible for epilogue injection. `semantic_dag::OpClass`
//! provides a richer classification derived from `OpTrace::ComputePattern`
//! and is preferred in new code paths.

use crate::compiler::graph::OpKind;
#[cfg(test)]
use crate::compiler::graph::SymDim;

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
    /// Attention computation (MHA, CachedGQA).
    /// Can fuse with preceding projections and following output projection.
    Attention,
    /// Opaque / metadata-only (Transpose, Reshape).
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
        OpKind::Silu | OpKind::Gelu | OpKind::Tanh | OpKind::Add | OpKind::Mul | OpKind::ScaleConst { .. } | OpKind::Residual | OpKind::LogitSoftcap { .. } => {
            OpSemantics::Elementwise
        }
        // Gated activations: elementwise (two inputs, one output, no reduction)
        OpKind::SwiGlu | OpKind::SwiGluClipped { .. } | OpKind::GeGlu => OpSemantics::Elementwise,

        // GEMM variants
        OpKind::Gemm { .. } | OpKind::GemmBias { .. } | OpKind::QuantGemm { .. } => {
            OpSemantics::Gemm
        }

        // Dequantize: elementwise decode
        OpKind::Dequantize { .. } => OpSemantics::Elementwise,

        // Reductions (need full input before output)
        OpKind::Softmax | OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } | OpKind::MeanPool { .. } | OpKind::L2Normalize { .. } | OpKind::QkNorm { .. } | OpKind::HeadRmsNorm { .. } | OpKind::ValueNorm { .. } => {
            OpSemantics::Reduction
        }

        // RoPE: elementwise rotation (x*cos - y*sin, x*sin + y*cos)
        OpKind::RoPE { .. } | OpKind::DualRoPE { .. } => OpSemantics::Elementwise,

        // Attention: can fuse with pre/post projections
        OpKind::MultiHeadAttention { .. } | OpKind::CachedGQA { .. } => {
            OpSemantics::Attention
        }

        // Gather: memory-bound indexed lookup, not fusable
        OpKind::Gather { .. } => OpSemantics::Opaque,
        // QuantGather: memory-bound indexed dequantize, not fusable
        OpKind::QuantGather { .. } => OpSemantics::Opaque,

        // ColumnSlice: memory-bound row-major column slice (row_stride changes, real copy)
        OpKind::ColumnSlice { .. } => OpSemantics::Opaque,

        // Opaque: metadata-only ops (zero cost)
        OpKind::Transpose { .. } | OpKind::Reshape { .. } | OpKind::SliceView { .. } => {
            OpSemantics::Opaque
        }

        // MoE ops
        OpKind::MoEGate { .. } => OpSemantics::Gemm,
        OpKind::TopK { .. } => OpSemantics::Reduction,
        OpKind::WeightedSum { .. } => OpSemantics::Elementwise,

        // Packed-expert MoE (gpt-oss-20b): composite op, Opaque (no fusion)。
        OpKind::MoERouter { .. }
        | OpKind::MoEDispatchPacked { .. } => OpSemantics::Opaque,

        // KV scatter write: pure memory op (no compute)
        OpKind::KvScatterWrite { .. } => OpSemantics::Opaque,

        // P4/P5 stub variants: treat as opaque for now
        OpKind::VariableLengthBatch
        | OpKind::AttentionSkipMask { .. }
        | OpKind::FusedRmsNormGemm { .. }
        | OpKind::ResidualWithTelemetry { .. }
        | OpKind::EntropyGate { .. }
        | OpKind::VRangeQuant { .. }
        | OpKind::KvCentroidPrefetch { .. }
        | OpKind::LayerBypass { .. }
        | OpKind::GateMask { .. }
        | OpKind::MaskedGemm { .. }
        | OpKind::MoEConditionalAdd { .. }
        | OpKind::SoftmaxWithEntropy { .. }
        | OpKind::MegaKernelDispatch { .. }
        // AltUpPredict/AltUpCorrect/AltUpInject: Injective composite ops (Gemma 4 E2B/E4B)
        | OpKind::AltUpPredict { .. }
        | OpKind::AltUpCorrect { .. }
        | OpKind::AltUpInject { .. }
        // DepthwiseConv1D: per-channel 1D conv,组合算子(USM Conformer),Opaque
        | OpKind::DepthwiseConv1D { .. }
        // PatchEmbed: Conv2D 滑动窗口 (SigLIP/ViT),Opaque 复合算子
        | OpKind::PatchEmbed { .. } => OpSemantics::Opaque,

        // LearnedPos2D: pure binary elementwise add (SigLIP/ViT)
        OpKind::LearnedPos2D { .. } => OpSemantics::Elementwise,

        // ARCH-SG-QTAP: Q-Tap is a pure side-effect store (ring buffer write + atomic bump).
        // Classify as Opaque so no fusion pass attempts to merge it with neighbours —
        // keeping the tap strictly post-GEMM preserves the semantics in SPEC §4.
        OpKind::QTapSTG { .. } => OpSemantics::Opaque,

        // Mega-kernel generate loop ops (GRAPH-SHAPE-DRIVEN-MEGA-KERNEL §2.2)
        OpKind::Argmax { .. } => OpSemantics::Reduction,
        OpKind::StoreToken | OpKind::CheckStopCondition => OpSemantics::Opaque,

        // Business config ops (§1.5 conditional graph construction): side-effect / control, Opaque
        OpKind::WriteLogits { .. }
        | OpKind::EarlyExit { .. }
        | OpKind::GuardrailCheck { .. }
        | OpKind::SgInject { .. }
        | OpKind::SgDetect { .. }
        | OpKind::CotStepCheck { .. }
        | OpKind::SessionKvRestore
        | OpKind::MmHiddenInject { .. }
        | OpKind::MtpDraft { .. } => OpSemantics::Opaque,

        // MLA (Multi-head Latent Attention) — DeepSeek V3/R1, Kimi-K2
        OpKind::MlaKvCompress { .. } | OpKind::MlaQAbsorb { .. } | OpKind::MlaVRestore { .. } => {
            OpSemantics::Gemm
        }
        OpKind::MlaAttention { .. } => OpSemantics::Attention,
        OpKind::MlaRopeMerge { .. } => OpSemantics::Elementwise,
    }
}

/// Determine the performance bottleneck for an operation.
///
/// Uses the roofline model: if arithmetic intensity > ridge point,
/// the op is compute-bound; otherwise memory-bound.
/// ARCH-DTYPE-FULLCHAIN-ORCH: `graph_dtype` used for QuantGemm activation/output byte estimation.
pub fn bottleneck(kind: &OpKind, graph_dtype: crate::types::DType) -> BottleneckType {
    match kind {
        // GEMM: compute-bound when K is large enough (typical for transformer layers)
        OpKind::Gemm { m, n, k, dtype, .. } | OpKind::GemmBias { m, n, k, dtype, .. } => {
            let m_val = m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model");
            let elem_bytes = dtype.size_bytes();
            let flops = 2 * m_val * n * k;
            let bytes = elem_bytes * (m_val * k + k * n + m_val * n);
            if bytes > 0 && (flops / bytes) >= 4 {
                BottleneckType::ComputeBound
            } else {
                BottleneckType::MemoryBound
            }
        }
        // Quantized GEMM: fewer bytes for weights, activation dtype from graph context
        OpKind::QuantGemm { m, n, k, quant_type } => {
            let m_val = m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model");
            let act_bytes = graph_dtype.size_bytes(); // activation/output dtype
            let bits = quant_type.bits() as usize;
            let weight_bytes = (k * n * bits + 7) / 8;
            let input_bytes = m_val * k * act_bytes;
            let output_bytes = m_val * n * act_bytes;
            let flops = 2 * m_val * n * k;
            let total_bytes = weight_bytes + input_bytes + output_bytes;
            if total_bytes > 0 && (flops / total_bytes) >= 4 {
                BottleneckType::ComputeBound
            } else {
                BottleneckType::MemoryBound
            }
        }
        // Dequantize: pure memory-bound
        OpKind::Dequantize { .. } => BottleneckType::MemoryBound,
        // QuantGather: indexed dequantize, memory-bound (read quant blocks → decode → write f32)
        OpKind::QuantGather { .. } => BottleneckType::MemoryBound,
        // Everything else is memory-bound (bandwidth-limited)
        _ => BottleneckType::MemoryBound,
    }
}

/// Estimate arithmetic intensity (FLOP / byte) for an operation.
///
/// Used by the roofline model to predict whether an op benefits from
/// fusion (memory-bound ops benefit most from eliminating round-trips).
/// ARCH-DTYPE-FULLCHAIN-ORCH: `graph_dtype` used for non-GEMM ops and QuantGemm activation bytes.
pub fn arithmetic_intensity(kind: &OpKind, graph_dtype: crate::types::DType) -> f64 {
    let eb = graph_dtype.size_bytes() as f64;
    match kind {
        OpKind::Gemm { m, n, k, dtype, .. } | OpKind::GemmBias { m, n, k, dtype, .. } => {
            let m_val = m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model");
            let elem_bytes = dtype.size_bytes() as f64;
            let flops = (2 * m_val * n * k) as f64;
            let bytes = elem_bytes * (m_val * k + k * n + m_val * n) as f64;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::QuantGemm { m, n, k, quant_type } => {
            let m_val = m.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model");
            let bits = quant_type.bits() as usize;
            let flops = (2 * m_val * n * k) as f64;
            let weight_bytes = ((k * n * bits + 7) / 8) as f64;
            let io_bytes = (m_val * k + m_val * n) as f64 * eb;
            if (weight_bytes + io_bytes) > 0.0 { flops / (weight_bytes + io_bytes) } else { 0.0 }
        }
        OpKind::Dequantize { num_elements, bits, .. } => {
            let flops = (*num_elements * 2) as f64;
            let input_bytes = *num_elements as f64 * (*bits as f64 / 8.0);
            let output_bytes = *num_elements as f64 * eb;
            let bytes = input_bytes + output_bytes;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::Silu | OpKind::Gelu | OpKind::Tanh | OpKind::LogitSoftcap { .. } => {
            // ~10 FLOPs per element, 2*eb bytes (read + write)
            10.0 / (2.0 * eb)
        }
        OpKind::Add | OpKind::Mul | OpKind::ScaleConst { .. } | OpKind::Residual => {
            // 1 FLOP, 3*eb bytes (2 reads + 1 write)
            1.0 / (3.0 * eb)
        }
        OpKind::SwiGlu | OpKind::GeGlu => {
            // ~12 FLOPs (silu/gelu + mul), 3*eb bytes (2 reads + 1 write)
            12.0 / (3.0 * eb)
        }
        OpKind::SwiGluClipped { .. } => {
            // ~16 FLOPs (4 clamps + silu + mul), 3*eb bytes (2 reads + 1 write).
            // Same memory profile as SwiGLU, marginally higher compute.
            16.0 / (3.0 * eb)
        }
        OpKind::Softmax => {
            // 3-pass: ~5N FLOPs, ~3*eb*N bytes
            5.0 / (3.0 * eb)
        }
        OpKind::RmsNorm { .. } | OpKind::LayerNorm { .. } | OpKind::L2Normalize { .. } | OpKind::QkNorm { .. } | OpKind::HeadRmsNorm { .. } | OpKind::ValueNorm { .. } => {
            // 2-pass: ~5N FLOPs, ~3*eb*N bytes
            5.0 / (3.0 * eb)
        }
        OpKind::RoPE { .. } | OpKind::DualRoPE { .. } => {
            // 6 FLOPs per pair, 4*eb bytes (2 reads + 2 writes) + cos/sin overhead
            6.0 / (4.0 * eb)
        }
        // Multi-head attention: O(s^2 * d * h) compute, dominated by QK^T and attn@V
        OpKind::MultiHeadAttention { seq_len, num_heads, num_kv_heads: _, head_dim, causal: _, attention_sinks: _ } => {
            let s = seq_len.max_for_allocation_strict().expect("ARCH-SYMDIM: Symbolic dim must have max_value in cost model") as f64;
            let h = *num_heads as f64;
            let d = *head_dim as f64;
            let hidden = h * d;
            // FLOPs: 2*s*s*d*h (QK^T) + 3*s*s*h (softmax) + 2*s*s*d*h (attn@V)
            let flops = 4.0 * s * s * d * h + 3.0 * s * s * h;
            // Bytes: 3*s*hidden*sizeof(f32) (Q,K,V read) + s*hidden*sizeof(f32) (output write)
            let f32_bytes = std::mem::size_of::<f32>() as f64;
            let bytes = 4.0 * s * hidden * f32_bytes;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::CachedGQA { seq_len, total_seq, num_heads, head_dim, kv_dtype, .. } => {
            let s = *seq_len as f64;
            let t = *total_seq as f64;
            let h = *num_heads as f64;
            let d = *head_dim as f64;
            let kv_elem = kv_dtype.size_bytes() as f64;
            let flops = 2.0 * s * t * d * h + 3.0 * s * t * h + 2.0 * s * t * d * h;
            // Q: f32, K/V: kv_dtype bytes, output: f32
            let f32_bytes = std::mem::size_of::<f32>() as f64;
            let bytes = s * h * d * f32_bytes + 2.0 * t * h * d * kv_elem + s * h * d * f32_bytes;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::MoEGate { seq_len, num_experts, hidden, .. } => {
            let flops = (2 * seq_len * num_experts * hidden) as f64;
            let bytes = (4 * (seq_len * hidden + hidden * num_experts + seq_len * num_experts)) as f64;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::TopK { .. } | OpKind::WeightedSum { .. } => {
            // Control-flow heavy, memory-bound
            1.0 / 8.0
        }
        OpKind::MeanPool { .. } | OpKind::Transpose { .. } | OpKind::Reshape { .. } | OpKind::SliceView { .. } => {
            // Pure data movement, 0 compute
            0.0
        }
        // Gather: memory-bound indexed copy, 0 FLOPs (pure memcpy)
        OpKind::Gather { .. } => 0.0,
        // QuantGather: memory-bound indexed dequantize, ~1 FLOP/elem (scale * nibble) — treat as 0
        OpKind::QuantGather { .. } => 0.0,
        // ColumnSlice: memory-bound row-major column slice, 0 FLOPs (pure copy)
        OpKind::ColumnSlice { .. } => 0.0,
        // KV scatter write: pure memory movement, 0 compute
        OpKind::KvScatterWrite { .. } => 0.0,
        // ARCH-SG-QTAP: pure side-effect memcpy + atomic — zero FLOPs, all memory.
        OpKind::QTapSTG { .. } => 0.0,
        // Mega-kernel generate loop ops (GRAPH-SHAPE-DRIVEN-MEGA-KERNEL §2.2)
        // Argmax: ~2 FLOPs/elem (compare + update max), memory-bound scan
        OpKind::Argmax { vocab_size } => {
            let v = *vocab_size as f64;
            let flops = 2.0 * v;
            let bytes = eb * (v + 1.0); // logits read + token_id write
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        // StoreToken / CheckStopCondition: pure memory/control, 0 compute
        OpKind::StoreToken | OpKind::CheckStopCondition => 0.0,
        // Business config ops (§1.5): side-effect / control, 0 compute
        OpKind::WriteLogits { .. }
        | OpKind::EarlyExit { .. }
        | OpKind::GuardrailCheck { .. }
        | OpKind::SgInject { .. }
        | OpKind::SgDetect { .. }
        | OpKind::CotStepCheck { .. }
        | OpKind::SessionKvRestore
        | OpKind::MmHiddenInject { .. }
        | OpKind::MtpDraft { .. } => 0.0,
        // P4/P5 stub variants: treat as memory-bound (0 compute)
        OpKind::VariableLengthBatch
        | OpKind::AttentionSkipMask { .. }
        | OpKind::FusedRmsNormGemm { .. }
        | OpKind::ResidualWithTelemetry { .. }
        | OpKind::EntropyGate { .. }
        | OpKind::VRangeQuant { .. }
        | OpKind::KvCentroidPrefetch { .. }
        | OpKind::LayerBypass { .. }
        | OpKind::GateMask { .. }
        | OpKind::MaskedGemm { .. }
        | OpKind::MoEConditionalAdd { .. }
        | OpKind::SoftmaxWithEntropy { .. }
        | OpKind::MegaKernelDispatch { .. } => 0.0,
        OpKind::AltUpPredict { seq_len, num_preds, hidden } => {
            let s = seq_len.max_for_allocation_strict()
                .expect("ARCH-SYMDIM: AltUpPredict seq_len must have max_value") as f64;
            let p = *num_preds as f64;
            let h = *hidden as f64;
            // predictions[p][s][h] = stacked[p][s][h] + Σ_q coefs[p][q] · stacked[q][s][h]
            // FLOPs ≈ 2 × P × P × s × h, Bytes ≈ eb × (P×s×h + s×P²)
            let flops = 2.0 * p * p * s * h;
            let bytes = eb * (p * s * h + s * p * p);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::AltUpCorrect { seq_len, num_preds, hidden } => {
            let s = seq_len.max_for_allocation_strict()
                .expect("ARCH-SYMDIM: AltUpCorrect seq_len must have max_value") as f64;
            let p = *num_preds as f64;
            let h = *hidden as f64;
            // corrected[p][s][h] = predictions[p][s][h] + corrected_coefs[s][p] × innovation[s][h]
            // FLOPs ≈ 2 × P × s × h, Bytes ≈ eb × (P×s×h + s×P + s×h)
            let flops = 2.0 * p * s * h;
            let bytes = eb * (p * s * h + s * p + s * h);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::AltUpInject { seq_len, num_preds, hidden } => {
            let s = seq_len.max_for_allocation_strict()
                .expect("ARCH-SYMDIM: AltUpInject seq_len must have max_value") as f64;
            let p = *num_preds as f64;
            let h = *hidden as f64;
            // corrected[p][s][h] += ple_projected[s][h] for p=1..P-1
            // FLOPs ≈ 2 × (P-1) × s × h, Bytes ≈ eb × (P×s×h + s×h)
            let flops = 2.0 * (p - 1.0) * s * h;
            let bytes = eb * (p * s * h + s * h);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        // DepthwiseConv1D: per-channel 1D conv,AI 仅与 kernel_size 相关 (独立 channel
        // 共享同一权重列),seq_len 外循环 → roofline 看作 memory-bound。
        // FLOPs = 2 × kernel × channels × seq (近似,不计 boundary pad),
        // Bytes ≈ eb × (2 × seq × channels + channels × kernel)。
        OpKind::DepthwiseConv1D { channels, kernel_size, .. } => {
            // 用典型 seq=128 作为 representative 估算 (roofline 近似)
            let s = 128.0_f64;
            let c = *channels as f64;
            let k = *kernel_size as f64;
            let flops = 2.0 * s * c * k;
            let bytes = eb * (2.0 * s * c + c * k);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        // PatchEmbed: Conv2D sliding window (SigLIP/ViT vision tower)。
        // FLOPs = 2 × num_patches × embed_dim × in_channels × patch_size²
        // Bytes ≈ eb × (image_elems + kernel_elems + patches_elems)
        //        = eb × (in_channels × image_size² + embed × in_channels × patch²
        //               + num_patches × embed)
        OpKind::PatchEmbed { patch_size, embed_dim, in_channels, image_size } => {
            let ps = *patch_size as f64;
            let ed = *embed_dim as f64;
            let ic = *in_channels as f64;
            let is_ = *image_size as f64;
            let num_patches_side = is_ / ps;
            let num_patches = num_patches_side * num_patches_side;
            let flops = 2.0 * num_patches * ed * ic * ps * ps;
            let image_elems = ic * is_ * is_;
            let kernel_elems = ed * ic * ps * ps;
            let patches_elems = num_patches * ed;
            let bytes = eb * (image_elems + kernel_elems + patches_elems);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        // LearnedPos2D: 1 FLOP per element, 3*eb bytes (2 reads + 1 write)
        OpKind::LearnedPos2D { .. } => {
            1.0 / (3.0 * eb)
        }
        // MoERouter: hidden @ weight.T + bias → softmax → top-k。
        OpKind::MoERouter { num_experts, top_k, hidden, seq_len } => {
            let s = seq_len.max_for_allocation_strict()
                .expect("ARCH-SYMDIM: MoERouter seq_len must have max_value") as f64;
            let e = *num_experts as f64;
            let h = *hidden as f64;
            let k = *top_k as f64;
            let flops = s * (2.0 * e * h + e + k); // GEMM + bias + softmax + top-k
            let io_bytes = eb * (s * h + e * h + s * k * 2.0);
            if io_bytes > 0.0 { flops / io_bytes } else { 0.0 }
        }
        // MoEDispatchPacked: composite packed-expert MoE。
        // 按选中的 top_k 个 expert 每 token 计算: gate_up_dequant + bias +
        // clipped SwiGLU + down_dequant + GEMV + accumulate。
        // FLOPs ≈ top_k × (4 · hidden · intermediate + 8 · intermediate)  per token。
        // Bytes ≈ top_k × (hidden · intermediate_size / 2 + intermediate_size)  weight
        //         + activation I/O。
        OpKind::MoEDispatchPacked { top_k, intermediate_size, hidden, seq_len, mxfp4_block_size, .. } => {
            let s = seq_len.max_for_allocation_strict()
                .expect("ARCH-SYMDIM: MoEDispatchPacked seq_len must have max_value") as f64;
            let k = *top_k as f64;
            let im = *intermediate_size as f64;
            let h = *hidden as f64;
            let flops = s * k * (4.0 * h * im + 8.0 * im);
            // mxfp4 weights: gate_up 2·im·h × 4 bits/8 + scales 2·im·h / block_size。
            let bs = *mxfp4_block_size as f64;
            let mxfp4_bytes = 2.0 * im * h * 0.5 + 2.0 * im * h / bs
                + im * h * 0.5 + im * h / bs;
            let io_bytes = eb * (s * h + s * h);
            let bytes = mxfp4_bytes + io_bytes;
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        // MLA GEMM variants: use same roofline model as standard GEMM
        OpKind::MlaKvCompress { m, d_c, hidden } => {
            let m_val = m.max_for_allocation_strict().expect("ARCH-SYMDIM: MlaKvCompress m must have max_value") as f64;
            let flops = 2.0 * m_val * (*d_c as f64) * (*hidden as f64);
            let bytes = eb * (m_val * (*hidden as f64) + (*hidden as f64) * (*d_c as f64) + m_val * (*d_c as f64));
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::MlaQAbsorb { seq_len, num_heads, head_dim, d_c } | OpKind::MlaVRestore { seq_len, num_heads, head_dim, d_c } => {
            let s = seq_len.max_for_allocation_strict().expect("ARCH-SYMDIM: MLA GEMM seq_len must have max_value") as f64;
            let h = *num_heads as f64;
            let d = *head_dim as f64;
            let dc = *d_c as f64;
            let flops = 2.0 * s * h * d * dc;
            let bytes = eb * (s * h * d + h * dc * d + s * h * dc);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::MlaAttention { seq_len, num_heads, head_dim, d_c, .. } => {
            let s = seq_len.max_for_allocation_strict().expect("ARCH-SYMDIM: MlaAttention seq_len must have max_value") as f64;
            let h = *num_heads as f64;
            let d = *head_dim as f64;
            let dc = *d_c as f64;
            let flops = 4.0 * s * s * dc * h + 3.0 * s * s * h + 2.0 * s * s * d * h;
            let bytes = eb * (s * h * dc + s * dc + s * h * d);
            if bytes > 0.0 { flops / bytes } else { 0.0 }
        }
        OpKind::MlaRopeMerge { .. } => {
            6.0 / (4.0 * eb)
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
        OpKind::Add | OpKind::Silu | OpKind::Gelu | OpKind::Tanh | OpKind::Residual
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DType;

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
            classify(&OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false }),
            OpSemantics::Gemm
        );
        assert_eq!(
            classify(&OpKind::GemmBias { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false }),
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
        let kind = OpKind::Gemm { m: SymDim::Concrete(128), n: 4096, k: 4096, dtype: DType::F32, trans_b: false };
        assert_eq!(bottleneck(&kind, DType::F32), BottleneckType::ComputeBound);
    }

    #[test]
    fn test_bottleneck_small_gemm() {
        // M=1 GEMV: memory-bound
        let kind = OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false };
        assert_eq!(bottleneck(&kind, DType::F32), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_bottleneck_elementwise() {
        assert_eq!(bottleneck(&OpKind::Silu, DType::F32), BottleneckType::MemoryBound);
        assert_eq!(bottleneck(&OpKind::Add, DType::F32), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_arithmetic_intensity_gemm() {
        let ai = arithmetic_intensity(&OpKind::Gemm { m: SymDim::Concrete(128), n: 4096, k: 4096, dtype: DType::F32, trans_b: false }, DType::F32);
        // Should be high (compute-bound)
        assert!(ai > 10.0, "GEMM AI={ai}, expected >10");
    }

    #[test]
    fn test_arithmetic_intensity_add() {
        let ai = arithmetic_intensity(&OpKind::Add, DType::F32);
        assert!(ai < 1.0, "Add AI={ai}, expected <1");
    }

    #[test]
    fn test_fusable_as_gemm_epilogue() {
        assert!(fusable_as_gemm_epilogue(&OpKind::Add));
        assert!(fusable_as_gemm_epilogue(&OpKind::Silu));
        assert!(!fusable_as_gemm_epilogue(&OpKind::SwiGlu));
        assert!(!fusable_as_gemm_epilogue(&OpKind::Softmax));
    }

    // ── QuantGemm / Dequantize tests ──

    #[test]
    fn test_quant_gemm_classify() {
        let kind = OpKind::QuantGemm { m: SymDim::Concrete(1), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 };
        assert_eq!(classify(&kind), OpSemantics::Gemm);
    }

    #[test]
    fn test_quant_gemm_bottleneck() {
        let kind = OpKind::QuantGemm { m: SymDim::Concrete(128), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 };
        assert_eq!(bottleneck(&kind, DType::F32), BottleneckType::ComputeBound);
    }

    #[test]
    fn test_quant_gemm_arithmetic_intensity() {
        let kind = OpKind::QuantGemm { m: SymDim::Concrete(128), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 };
        let ai = arithmetic_intensity(&kind, DType::F32);
        // Q4 weights are 4x smaller → higher AI than F32 GEMM
        assert!(ai > 20.0, "Q4 GEMM AI={ai}, expected >20");
    }

    #[test]
    fn test_dequantize_classify() {
        let kind = OpKind::Dequantize { num_elements: 4096, block_size: 32, bits: 4 };
        assert_eq!(classify(&kind), OpSemantics::Elementwise);
    }

    #[test]
    fn test_dequantize_bottleneck() {
        let kind = OpKind::Dequantize { num_elements: 4096, block_size: 32, bits: 4 };
        assert_eq!(bottleneck(&kind, DType::F32), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_classify_rope_elementwise() {
        assert_eq!(
            classify(&OpKind::RoPE { num_heads: 32, head_dim: 128, theta: 10000.0, partial: 1.0, rope_scaling: None }),
            OpSemantics::Elementwise,
        );
    }

    #[test]
    fn test_classify_attention() {
        use crate::compiler::graph::AttentionStrategy;
        assert_eq!(
            classify(&OpKind::MultiHeadAttention { seq_len: SymDim::Concrete(32), num_heads: 32, num_kv_heads: 32, head_dim: 128, causal: true, attention_sinks: false }),
            OpSemantics::Attention,
        );
        assert_eq!(
            classify(&OpKind::CachedGQA {
                seq_len: 1, total_seq: 128, num_heads: 32,
                num_kv_heads: 8, head_dim: 128,
                strategy: AttentionStrategy::Naive, kv_dtype: DType::F32,
            }),
            OpSemantics::Attention,
        );
    }

    #[test]
    fn test_classify_opaque_only_metadata() {
        assert_eq!(classify(&OpKind::Transpose { perm: vec![0, 2, 1, 3] }), OpSemantics::Opaque);
        assert_eq!(classify(&OpKind::Reshape { target_shape: vec![1, 4096] }), OpSemantics::Opaque);
    }

    // ── Additional tests for semantic classification coverage ──

    #[test]
    fn test_classify_moe_gate_as_gemm() {
        // MoE gate is semantically a GEMM (hidden @ router_weight)
        let kind = OpKind::MoEGate { seq_len: 1, num_experts: 8, hidden: 4096, top_k: 2 };
        assert_eq!(classify(&kind), OpSemantics::Gemm);
    }

    #[test]
    fn test_classify_reduction_ops() {
        // Multiple reduction-classified ops: LayerNorm, MeanPool, L2Normalize, QkNorm, Argmax
        assert_eq!(classify(&OpKind::LayerNorm { eps: 1e-5 }), OpSemantics::Reduction);
        assert_eq!(
            classify(&OpKind::MeanPool { seq_len: 128, hidden: 768, cls_mode: false }),
            OpSemantics::Reduction,
        );
        assert_eq!(
            classify(&OpKind::L2Normalize { hidden: 768 }),
            OpSemantics::Reduction,
        );
        assert_eq!(
            classify(&OpKind::QkNorm { head_dim: 128, eps: 1e-6 }),
            OpSemantics::Reduction,
        );
        assert_eq!(
            classify(&OpKind::Argmax { vocab_size: 32000 }),
            OpSemantics::Reduction,
        );
    }

    #[test]
    fn test_classify_mla_variants() {
        // MLA GEMM variants should classify as Gemm
        assert_eq!(
            classify(&OpKind::MlaKvCompress { m: SymDim::Concrete(1), d_c: 512, hidden: 4096 }),
            OpSemantics::Gemm,
        );
        assert_eq!(
            classify(&OpKind::MlaQAbsorb { seq_len: SymDim::Concrete(1), num_heads: 32, head_dim: 128, d_c: 512 }),
            OpSemantics::Gemm,
        );
        assert_eq!(
            classify(&OpKind::MlaVRestore { seq_len: SymDim::Concrete(1), num_heads: 32, head_dim: 128, d_c: 512 }),
            OpSemantics::Gemm,
        );
        // MLA Attention should classify as Attention
        assert_eq!(
            classify(&OpKind::MlaAttention { seq_len: SymDim::Concrete(32), num_heads: 32, head_dim: 128, d_c: 512, d_rope: 64, causal: true }),
            OpSemantics::Attention,
        );
        // MLA RoPE merge is elementwise
        assert_eq!(
            classify(&OpKind::MlaRopeMerge { seq_len: SymDim::Concrete(1), d_c: 512, d_rope: 64 }),
            OpSemantics::Elementwise,
        );
    }

    #[test]
    fn test_arithmetic_intensity_elementwise_ops_low() {
        // Elementwise ops should have low arithmetic intensity (memory-bound)
        let silu_ai = arithmetic_intensity(&OpKind::Silu, DType::F32);
        let add_ai = arithmetic_intensity(&OpKind::Add, DType::F32);
        let mul_ai = arithmetic_intensity(&OpKind::Mul, DType::F32);
        let residual_ai = arithmetic_intensity(&OpKind::Residual, DType::F32);

        assert!(silu_ai > 0.0, "Silu AI should be positive, got {silu_ai}");
        assert!(add_ai > 0.0, "Add AI should be positive, got {add_ai}");
        assert!(mul_ai > 0.0, "Mul AI should be positive, got {mul_ai}");
        assert!(residual_ai > 0.0, "Residual AI should be positive, got {residual_ai}");

        // All should be below 2.0 (memory-bound)
        assert!(silu_ai < 2.0, "Silu AI={silu_ai}, expected < 2.0");
        assert!(add_ai < 2.0, "Add AI={add_ai}, expected < 2.0");
        assert!(mul_ai < 2.0, "Mul AI={mul_ai}, expected < 2.0");
        assert!(residual_ai < 2.0, "Residual AI={residual_ai}, expected < 2.0");
    }

    #[test]
    fn test_arithmetic_intensity_quant_gemm_higher_than_f32() {
        // QuantGemm with 4-bit weights should have higher AI than F32 GEMM (smaller weights)
        let f32_gemv = arithmetic_intensity(
            &OpKind::Gemm { m: SymDim::Concrete(1), n: 4096, k: 4096, dtype: DType::F32, trans_b: false },
            DType::F32,
        );
        let q4_gemv = arithmetic_intensity(
            &OpKind::QuantGemm { m: SymDim::Concrete(1), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 },
            DType::F32,
        );
        assert!(q4_gemv > f32_gemv, "Q4 GEMM AI={q4_gemv} should be > F32 GEMM AI={f32_gemv}");
    }

    #[test]
    fn test_bottleneck_quant_gemm_memory_bound_small() {
        // Small M=1 QuantGemm should be memory-bound (GEMV)
        let kind = OpKind::QuantGemm { m: SymDim::Concrete(1), n: 4096, k: 4096, quant_type: crate::quant::QuantType::Q4_0 };
        assert_eq!(bottleneck(&kind, DType::F32), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_arithmetic_intensity_zero_for_pure_memory_ops() {
        // Pure memory/data-movement ops should have zero arithmetic intensity
        assert_eq!(arithmetic_intensity(&OpKind::Gather { table_rows: 32000, embed_dim: 4096, index_dim: SymDim::Concrete(1), indices_kind: crate::compiler::graph::GatherIndicesKind::Tensor, scale: None }, DType::F32), 0.0);
        assert_eq!(arithmetic_intensity(&OpKind::Reshape { target_shape: vec![1, 4096] }, DType::F32), 0.0);
        assert_eq!(arithmetic_intensity(&OpKind::Transpose { perm: vec![0, 2, 1, 3] }, DType::F32), 0.0);
    }

    #[test]
    fn test_fusable_epilogue_excludes_non_elementwise() {
        // Ops that are not fusable as GEMM epilogue (Gelu IS fusable, so excluded)
        assert!(!fusable_as_gemm_epilogue(&OpKind::Softmax));
        assert!(!fusable_as_gemm_epilogue(&OpKind::RmsNorm { eps: 1e-5 }));
        assert!(!fusable_as_gemm_epilogue(&OpKind::Mul));
        assert!(!fusable_as_gemm_epilogue(&OpKind::SwiGlu));
        assert!(!fusable_as_gemm_epilogue(&OpKind::RoPE { num_heads: 32, head_dim: 128, theta: 10000.0, partial: 1.0, rope_scaling: None }));
    }

    #[test]
    fn test_classify_gated_activations_as_elementwise() {
        // All gated activations (SwiGlu, GeGlu, clipped SwiGlu) are elementwise
        assert_eq!(classify(&OpKind::SwiGlu), OpSemantics::Elementwise);
        assert_eq!(classify(&OpKind::GeGlu), OpSemantics::Elementwise);
        assert_eq!(classify(&OpKind::SwiGluClipped { limit: 7.0 }), OpSemantics::Elementwise);
    }

    #[test]
    fn test_classify_moe_router_and_dispatch_as_opaque() {
        // MoE router and dispatch packed are composite opaque ops
        assert_eq!(
            classify(&OpKind::MoERouter { num_experts: 64, top_k: 8, hidden: 4096, seq_len: SymDim::Concrete(1) }),
            OpSemantics::Opaque,
        );
        assert_eq!(
            classify(&OpKind::MoEDispatchPacked {
                num_experts: 64, top_k: 8, mxfp4_block_size: 32,
                swiglu_limit: 7.0, intermediate_size: 1408, hidden: 4096,
                seq_len: SymDim::Concrete(1),
            }),
            OpSemantics::Opaque,
        );
    }

}
