//! QuantPipeline — end-to-end quantized inference pipeline integration.
//!
//! REQ-QPJ-005: Unified entry point that wires together the full quantized
//! inference path: quantized weight loading → dequantize → GEMM → output.
//!
//! ## Architecture
//!
//! ```text
//! QuantPipeline
//!   ├── QuantGather path:  quantized embed table → block decode → F32 embedding rows
//!   │     TraceOp::QuantGather → auto_select → emit_quant_gather_inline → VmInstr
//!   │
//!   └── QuantGemm path:    quantized weight matrix → tiled dequant + FMA → F32 output
//!         TraceOp::QuantGemm → auto_select → emit_quant_gemm_inline → VmInstr
//! ```
//!
//! Both paths go through the complete SymExec → auto_select → lowering pipeline.
//! No per-format hand-written VmInstr emission; all dequantization is driven by
//! `QuantFormatDescriptor` metadata via `DecodeTraceBuilder` trace templates.

use crate::compiler::codegen::vm::auto_select;
use crate::compiler::codegen::vm::instr::{
    BoundExpr, SimdWidth, VRegId, VmProgram,
};
use crate::compiler::trace::{
    self, QuantPrecision, TraceOp, TypedSlot,
};
use crate::quant::QuantType;
use crate::types::CompilerError;

// ────────────────────────────────────────────────────────────────────────────
// QuantPipeline: configuration + pipeline builder
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for a quantized operation pipeline.
///
/// Encapsulates all parameters needed to build and emit a complete quantized
/// inference operation (QuantGather or QuantGemm) through the JIT pipeline.
#[derive(Debug, Clone)]
pub struct QuantPipeline {
    /// Quantization format driving block decode strategy.
    pub quant_type: QuantType,
    /// SIMD width for the target architecture.
    pub width: SimdWidth,
    /// Default dtype for computation (typically F32 for dequantized values).
    pub compute_dtype: QuantPrecision,
}

impl QuantPipeline {
    /// Create a new QuantPipeline for the given quantization format.
    ///
    /// The SIMD width and compute dtype are derived from the quantization
    /// format's natural output precision (F32 for all GGUF block formats).
    pub fn new(quant_type: QuantType) -> Self {
        Self {
            quant_type,
            width: SimdWidth::W256,
            compute_dtype: QuantPrecision::F32,
        }
    }

    /// Create a QuantPipeline with explicit SIMD width.
    pub fn with_width(mut self, width: SimdWidth) -> Self {
        self.width = width;
        self
    }

    /// Create a QuantPipeline with explicit compute dtype.
    pub fn with_compute_dtype(mut self, dtype: QuantPrecision) -> Self {
        self.compute_dtype = dtype;
        self
    }

    // ── QuantGather path ──────────────────────────────────────────────

    /// Build trace template for a QuantGather operation.
    ///
    /// Generates the TraceOp sequence for quantized embedding lookup:
    /// Input → QuantGather {indices_ptr, embed_ptr, output_ptr, quant_type, ...}
    ///
    /// The trace is designed to be consumed by `auto_select::dispatch_trace_op`,
    /// which expands `TraceOp::QuantGather` into the full loop nest with
    /// `DecodeTraceBuilder`-driven block decode.
    pub fn build_gather_trace(
        &self,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Vec<TraceOp> {
        trace::build_quant_gather_trace(self.quant_type, vocab_size, hidden_dim)
    }

    /// Emit a QuantGather operation directly into a VmProgram.
    ///
    /// This is the complete end-to-end path:
    /// 1. Build the QuantGather trace template
    /// 2. Propagate dtype through the trace
    /// 3. Lower via `auto_lower_trace_typed` → VmInstr sequence
    ///
    /// The result is inline VmInstrs in `prog` that perform:
    ///   seq loop → load token_id → compute row base → block loop →
    ///   DecodeTraceBuilder decode → auto_lower_trace_raw → VecStore
    pub fn emit_gather(
        &self,
        prog: &mut VmProgram,
        indices_ptr: VRegId,
        embed_ptr: VRegId,
        output_ptr: VRegId,
        vocab_size: usize,
        hidden_dim: usize,
        seq_bound: BoundExpr,
    ) -> Result<(), CompilerError> {
        // Delegate to the trace-driven emit path used by auto_select.
        // This ensures QuantGather goes through SymExec → auto_select → lowering.
        super::codegen::vm::quant_gather_emit::emit_quant_gather_inline(
            prog,
            seq_bound,
            vocab_size,
            hidden_dim,
            self.quant_type,
            self.width,
            indices_ptr,
            embed_ptr,
            output_ptr,
            self.compute_dtype,
        )
    }

    // ── QuantGemm path ────────────────────────────────────────────────

    /// Build trace template for a QuantGemm operation.
    ///
    /// Generates the TraceOp sequence for quantized matrix multiplication:
    /// Input → QuantGemm {input_ptr, weight_ptr, output_ptr, quant_type, m, n, k}
    ///
    /// The trace is designed to be consumed by `auto_select::dispatch_trace_op`,
    /// which expands `TraceOp::QuantGemm` into the tiled loop nest with
    /// block decode + FMA accumulation.
    pub fn build_gemm_trace(
        &self,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<TraceOp> {
        trace::build_quant_gemm_trace(self.quant_type, m, n, k)
    }

    /// Emit a QuantGemm operation directly into a VmProgram.
    ///
    /// Complete end-to-end path:
    /// 1. Build the QuantGemm trace template
    /// 2. Propagate dtype through the trace
    /// 3. Lower via `auto_lower_trace_typed` → VmInstr sequence
    ///
    /// The result is inline VmInstrs in `prog` that perform:
    ///   M×N×K tiled loop → per-tile block decode → DotProduct/FMA → store
    pub fn emit_gemm(
        &self,
        prog: &mut VmProgram,
        input_ptr: VRegId,
        weight_ptr: VRegId,
        output_ptr: VRegId,
        m: usize,
        n: usize,
        k: usize,
        m_bound: BoundExpr,
    ) -> Result<(), CompilerError> {
        super::codegen::vm::moe_quant_emit::emit_quant_gemm_inline(
            prog,
            m_bound,
            n,
            k,
            self.quant_type,
            self.width,
            input_ptr,
            weight_ptr,
            output_ptr,
            self.compute_dtype,
            crate::dispatch::device_profile::DotProductCap::SimdAssisted,
        )
    }

    // ── Unified trace-driven path ─────────────────────────────────────

    /// Lower a pre-built trace through the full JIT pipeline.
    ///
    /// Takes a trace (from `build_gather_trace` or `build_gemm_trace`),
    /// allocates typed input slots, propagates dtype, and lowers to VmInstrs.
    ///
    /// This is the generic path used when the caller wants to manipulate
    /// the trace before lowering (e.g., fusing additional ops).
    pub fn lower_trace(
        &self,
        prog: &mut VmProgram,
        trace: &[TraceOp],
        input_vregs: &[VRegId],
    ) -> Result<Vec<TypedSlot>, CompilerError> {
        let typed_inputs: Vec<TypedSlot> = input_vregs
            .iter()
            .map(|&vreg| TypedSlot {
                vreg,
                dtype: self.compute_dtype,
            })
            .collect();
        auto_select::auto_lower_trace_typed(prog, trace, &typed_inputs, self.width)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// QuantPipelineBuilder: fluent API for constructing quantized inference ops
// ────────────────────────────────────────────────────────────────────────────

/// Fluent builder for constructing QuantPipeline operations.
///
/// Usage:
/// ```ignore
/// let pipeline = QuantPipelineBuilder::new(QuantType::Q4_0)
///     .width(SimdWidth::Avx2)
///     .build();
/// let trace = pipeline.build_gemm_trace(m, n, k);
/// pipeline.lower_trace(&mut prog, &trace, &input_vregs)?;
/// ```
pub struct QuantPipelineBuilder {
    quant_type: QuantType,
    width: Option<SimdWidth>,
    compute_dtype: Option<QuantPrecision>,
}

impl QuantPipelineBuilder {
    /// Start building a QuantPipeline for the given quantization format.
    pub fn new(quant_type: QuantType) -> Self {
        Self {
            quant_type,
            width: None,
            compute_dtype: None,
        }
    }

    /// Set SIMD width (default: Avx2).
    pub fn width(mut self, width: SimdWidth) -> Self {
        self.width = Some(width);
        self
    }

    /// Set compute dtype (default: F32).
    pub fn compute_dtype(mut self, dtype: QuantPrecision) -> Self {
        self.compute_dtype = Some(dtype);
        self
    }

    /// Build the QuantPipeline.
    pub fn build(self) -> QuantPipeline {
        let mut pipeline = QuantPipeline::new(self.quant_type);
        if let Some(w) = self.width {
            pipeline.width = w;
        }
        if let Some(dt) = self.compute_dtype {
            pipeline.compute_dtype = dt;
        }
        pipeline
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Integration helpers: wire QuantPipeline into the compilation pipeline
// ────────────────────────────────────────────────────────────────────────────

/// Generate the full trace for a quantized layer forward pass.
///
/// This function produces the complete TraceOp sequence for one transformer
/// layer's quantized GEMM operations (QKV projection + output projection +
/// FFN up/down/gate projections), each as a `TraceOp::QuantGemm`.
///
/// The returned traces are meant to be registered via `ScalarOpRegistry`
/// and consumed by the standard `SemanticDAG → FusionPlan → plan_lower` pipeline.
pub fn build_quant_layer_traces(
    quant_type: QuantType,
    hidden: usize,
    intermediate: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> QuantLayerTraces {
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;

    // QKV projection: [hidden, q_dim + 2*kv_dim]
    let qkv_k = hidden;
    let qkv_n = q_dim + 2 * kv_dim;

    // Output projection: [hidden, hidden]
    let out_k = hidden;
    let out_n = hidden;

    // FFN up projection: [hidden, intermediate]
    let ffn_up_k = hidden;
    let ffn_up_n = intermediate;

    // FFN down projection: [intermediate, hidden]
    let ffn_down_k = intermediate;
    let ffn_down_n = hidden;

    // FFN gate projection (SwiGLU/GeGLU): [hidden, intermediate]
    let ffn_gate_k = hidden;
    let ffn_gate_n = intermediate;

    let pipeline = QuantPipeline::new(quant_type);

    QuantLayerTraces {
        qkv_trace: pipeline.build_gemm_trace(0, qkv_n, qkv_k),
        qkv_dims: (0, qkv_n, qkv_k),
        output_trace: pipeline.build_gemm_trace(0, out_n, out_k),
        output_dims: (0, out_n, out_k),
        ffn_up_trace: pipeline.build_gemm_trace(0, ffn_up_n, ffn_up_k),
        ffn_up_dims: (0, ffn_up_n, ffn_up_k),
        ffn_down_trace: pipeline.build_gemm_trace(0, ffn_down_n, ffn_down_k),
        ffn_down_dims: (0, ffn_down_n, ffn_down_k),
        ffn_gate_trace: pipeline.build_gemm_trace(0, ffn_gate_n, ffn_gate_k),
        ffn_gate_dims: (0, ffn_gate_n, ffn_gate_k),
        quant_type,
    }
}

/// Trace templates for all quantized GEMM ops in one transformer layer.
///
/// Produced by `build_quant_layer_traces`, consumed by the compilation pipeline.
#[derive(Debug, Clone)]
pub struct QuantLayerTraces {
    /// QKV projection trace.
    pub qkv_trace: Vec<TraceOp>,
    /// QKV GEMM dimensions: (M, N, K).
    pub qkv_dims: (usize, usize, usize),

    /// Output projection trace.
    pub output_trace: Vec<TraceOp>,
    /// Output GEMM dimensions: (M, N, K).
    pub output_dims: (usize, usize, usize),

    /// FFN up projection trace.
    pub ffn_up_trace: Vec<TraceOp>,
    /// FFN up GEMM dimensions: (M, N, K).
    pub ffn_up_dims: (usize, usize, usize),

    /// FFN down projection trace.
    pub ffn_down_trace: Vec<TraceOp>,
    /// FFN down GEMM dimensions: (M, N, K).
    pub ffn_down_dims: (usize, usize, usize),

    /// FFN gate projection trace (SwiGLU/GeGLU models).
    pub ffn_gate_trace: Vec<TraceOp>,
    /// FFN gate GEMM dimensions: (M, N, K).
    pub ffn_gate_dims: (usize, usize, usize),

    /// Quantization format.
    pub quant_type: QuantType,
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_pipeline_new() {
        let pipeline = QuantPipeline::new(QuantType::Q4_0);
        assert!(matches!(pipeline.quant_type, QuantType::Q4_0));
        assert_eq!(pipeline.compute_dtype, QuantPrecision::F32);
    }

    #[test]
    fn test_quant_pipeline_builder() {
        let pipeline = QuantPipelineBuilder::new(QuantType::Q8_0)
            .width(SimdWidth::W256)
            .compute_dtype(QuantPrecision::F32)
            .build();
        assert!(matches!(pipeline.quant_type, QuantType::Q8_0));
    }

    #[test]
    fn test_build_gather_trace_structure() {
        let pipeline = QuantPipeline::new(QuantType::Q4_0);
        let trace = pipeline.build_gather_trace(32000, 4096);
        // build_quant_gather_trace produces: Input, Input, Input, QuantGather
        assert!(trace.len() >= 4, "gather trace should have at least 4 ops, got {}", trace.len());

        let has_gather = trace.iter().any(|op| matches!(op, TraceOp::QuantGather { .. }));
        assert!(has_gather, "gather trace should contain QuantGather op");
    }

    #[test]
    fn test_build_gemm_trace_structure() {
        let pipeline = QuantPipeline::new(QuantType::Q4_0);
        let trace = pipeline.build_gemm_trace(1, 4096, 4096);
        assert!(trace.len() >= 4, "gemm trace should have at least 4 ops, got {}", trace.len());

        let has_gemm = trace.iter().any(|op| matches!(op, TraceOp::QuantGemm { .. }));
        assert!(has_gemm, "gemm trace should contain QuantGemm op");
    }

    #[test]
    fn test_build_quant_layer_traces() {
        let traces = build_quant_layer_traces(
            QuantType::Q4_0,
            4096,   // hidden
            11008,  // intermediate
            32,     // num_heads
            32,     // num_kv_heads
            128,    // head_dim
        );
        assert!(matches!(traces.quant_type, QuantType::Q4_0));
        assert!(traces.qkv_trace.len() >= 4);
        assert!(traces.output_trace.len() >= 4);
        assert!(traces.ffn_up_trace.len() >= 4);
        assert!(traces.ffn_down_trace.len() >= 4);
        assert!(traces.ffn_gate_trace.len() >= 4);

        // QKV: N = q_dim + 2*kv_dim = 32*128 + 2*32*128 = 4096 + 8192 = 12288
        assert_eq!(traces.qkv_dims.1, 12288);
        assert_eq!(traces.qkv_dims.2, 4096);
    }

    #[test]
    fn test_quant_pipeline_various_formats() {
        for qt in [QuantType::Q4_0, QuantType::Q4_1, QuantType::Q8_0, QuantType::Q8_1] {
            let pipeline = QuantPipeline::new(qt);
            let gather = pipeline.build_gather_trace(32000, 4096);
            assert!(gather.iter().any(|op| matches!(op, TraceOp::QuantGather { .. })),
                "QuantGather trace missing for {:?}", qt);

            let gemm = pipeline.build_gemm_trace(1, 4096, 4096);
            assert!(gemm.iter().any(|op| matches!(op, TraceOp::QuantGemm { .. })),
                "QuantGemm trace missing for {:?}", qt);
        }
    }

    // ── 13 new tests ──────────────────────────────────────────────────────

    // @trace TEST-QP-07 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_pipeline_new_default_width_is_w256() {
        // Arrange: create pipeline with Q4_0
        let pipeline = QuantPipeline::new(QuantType::Q4_0);

        // Act & Assert: default SIMD width must be W256
        assert_eq!(pipeline.width, SimdWidth::W256,
            "QuantPipeline::new() should default to SimdWidth::W256");
    }

    // @trace TEST-QP-08 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_pipeline_with_width_overrides_default() {
        // Arrange: create pipeline and override width to W512
        let pipeline = QuantPipeline::new(QuantType::Q8_0)
            .with_width(SimdWidth::W512);

        // Act & Assert: width should be overridden
        assert_eq!(pipeline.width, SimdWidth::W512);
        assert!(matches!(pipeline.quant_type, QuantType::Q8_0));
        assert_eq!(pipeline.compute_dtype, QuantPrecision::F32);
    }

    // @trace TEST-QP-09 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_pipeline_with_compute_dtype_override() {
        // Arrange: create pipeline and override compute dtype
        let pipeline = QuantPipeline::new(QuantType::Q4_0)
            .with_compute_dtype(QuantPrecision::BF16);

        // Act & Assert: compute dtype should be overridden while quant_type remains
        assert_eq!(pipeline.compute_dtype, QuantPrecision::BF16);
        assert!(matches!(pipeline.quant_type, QuantType::Q4_0));
    }

    // @trace TEST-QP-10 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_builder_chains_all_fields() {
        // Arrange: use builder with all fields overridden
        let pipeline = QuantPipelineBuilder::new(QuantType::Q4K)
            .width(SimdWidth::W128)
            .compute_dtype(QuantPrecision::F16)
            .build();

        // Act & Assert: all builder fields should be applied
        assert!(matches!(pipeline.quant_type, QuantType::Q4K));
        assert_eq!(pipeline.width, SimdWidth::W128);
        assert_eq!(pipeline.compute_dtype, QuantPrecision::F16);
    }

    // @trace TEST-QP-11 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_builder_defaults_when_no_overrides() {
        // Arrange: use builder with no optional overrides
        let pipeline = QuantPipelineBuilder::new(QuantType::Q5_0).build();

        // Act & Assert: should match QuantPipeline::new() defaults
        assert!(matches!(pipeline.quant_type, QuantType::Q5_0));
        assert_eq!(pipeline.width, SimdWidth::W256);
        assert_eq!(pipeline.compute_dtype, QuantPrecision::F32);
    }

    // @trace TEST-QP-12 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_gather_trace_input_order_and_params() {
        // Arrange
        let pipeline = QuantPipeline::new(QuantType::Q5_1);
        let vocab = 50257;
        let hidden = 768;

        // Act
        let trace = pipeline.build_gather_trace(vocab, hidden);

        // Assert: exact 4 ops with correct ordering and parameters
        assert_eq!(trace.len(), 4, "gather trace should have exactly 4 ops");
        assert!(matches!(trace[0], TraceOp::Input(0)), "first op must be Input(0) for indices_ptr");
        assert!(matches!(trace[1], TraceOp::Input(1)), "second op must be Input(1) for embed_ptr");
        assert!(matches!(trace[2], TraceOp::Input(2)), "third op must be Input(2) for output_ptr");

        match &trace[3] {
            TraceOp::QuantGather { quant_type, vocab_size, hidden_dim } => {
                assert!(matches!(quant_type, QuantType::Q5_1));
                assert_eq!(*vocab_size, vocab);
                assert_eq!(*hidden_dim, hidden);
            }
            other => panic!("expected QuantGather, got {:?}", other),
        }
    }

    // @trace TEST-QP-13 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_gemm_trace_input_order_and_params() {
        // Arrange
        let pipeline = QuantPipeline::new(QuantType::Q8_1);
        let m = 2;
        let n = 512;
        let k = 1024;

        // Act
        let trace = pipeline.build_gemm_trace(m, n, k);

        // Assert: exact 4 ops with correct ordering and parameters
        assert_eq!(trace.len(), 4, "gemm trace should have exactly 4 ops");
        assert!(matches!(trace[0], TraceOp::Input(0)), "first op must be Input(0) for input_ptr");
        assert!(matches!(trace[1], TraceOp::Input(1)), "second op must be Input(1) for weight_ptr");
        assert!(matches!(trace[2], TraceOp::Input(2)), "third op must be Input(2) for output_ptr");

        match &trace[3] {
            TraceOp::QuantGemm { quant_type, m: m_val, n: n_val, k: k_val } => {
                assert!(matches!(quant_type, QuantType::Q8_1));
                assert_eq!(*m_val, m);
                assert_eq!(*n_val, n);
                assert_eq!(*k_val, k);
            }
            other => panic!("expected QuantGemm, got {:?}", other),
        }
    }

    // @trace TEST-QP-14 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_layer_traces_gqa_dimensions() {
        // Arrange: GQA model with num_heads=32, num_kv_heads=8, head_dim=128
        // kv_dim = 8 * 128 = 1024, q_dim = 32 * 128 = 4096
        let traces = build_quant_layer_traces(
            QuantType::Q4K,
            4096,   // hidden
            11008,  // intermediate
            32,     // num_heads
            8,      // num_kv_heads (GQA: fewer KV heads)
            128,    // head_dim
        );

        // Act & Assert: QKV N = q_dim + 2*kv_dim = 4096 + 2048 = 6144
        assert_eq!(traces.qkv_dims, (0, 6144, 4096));
        // Output projection: N = hidden = 4096, K = hidden = 4096
        assert_eq!(traces.output_dims, (0, 4096, 4096));
        // FFN up: N = intermediate = 11008, K = hidden = 4096
        assert_eq!(traces.ffn_up_dims, (0, 11008, 4096));
        // FFN down: N = hidden = 4096, K = intermediate = 11008
        assert_eq!(traces.ffn_down_dims, (0, 4096, 11008));
        // FFN gate: N = intermediate = 11008, K = hidden = 4096
        assert_eq!(traces.ffn_gate_dims, (0, 11008, 4096));
        // M dimension is always 0 (symbolic, bound at runtime)
        assert_eq!(traces.qkv_dims.0, 0);
        assert_eq!(traces.output_dims.0, 0);
    }

    // @trace TEST-QP-15 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_layer_traces_m_dim_is_zero_symbolic() {
        // Arrange: any quant type and dimensions
        let traces = build_quant_layer_traces(
            QuantType::Q8K,
            2048,
            5632,
            16,
            16,
            64,
        );

        // Act & Assert: all M dims must be 0 (symbolic, bound at runtime)
        assert_eq!(traces.qkv_dims.0, 0);
        assert_eq!(traces.output_dims.0, 0);
        assert_eq!(traces.ffn_up_dims.0, 0);
        assert_eq!(traces.ffn_down_dims.0, 0);
        assert_eq!(traces.ffn_gate_dims.0, 0);
    }

    // @trace TEST-QP-16 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_gather_trace_k_quant_and_iq_formats() {
        // Arrange: K-quant and IQ family formats
        let k_quant_formats = [
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K,
            QuantType::Q5K, QuantType::Q6K, QuantType::Q8K,
        ];
        let iq_formats = [
            QuantType::IQ3XXS, QuantType::IQ4NL,
        ];

        // Act & Assert: all K-quant and IQ formats should produce valid gather traces
        for qt in k_quant_formats.iter().chain(iq_formats.iter()) {
            let pipeline = QuantPipeline::new(*qt);
            let trace = pipeline.build_gather_trace(32000, 4096);
            assert_eq!(trace.len(), 4, "gather trace for {:?} should have 4 ops", qt);

            let gather_op = trace.iter().find(|op| matches!(op, TraceOp::QuantGather { .. }));
            assert!(gather_op.is_some(), "QuantGather missing for {:?}", qt);
        }
    }

    // @trace TEST-QP-17 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_gemm_trace_external_formats() {
        // Arrange: external quantization formats (AWQ, GPTQ, SqueezeLLM, NVFP4, MXFP4)
        let formats = [
            QuantType::AWQ4,
            QuantType::GPTQ4,
            QuantType::Squeeze,
            QuantType::Nvfp4,
            QuantType::Mxfp4 { block_size: 32 },
        ];

        // Act & Assert: all external formats should produce valid gemm traces
        for qt in &formats {
            let pipeline = QuantPipeline::new(*qt);
            let trace = pipeline.build_gemm_trace(1, 4096, 4096);
            assert_eq!(trace.len(), 4, "gemm trace for {:?} should have 4 ops", qt);

            match &trace[3] {
                TraceOp::QuantGemm { quant_type, .. } => {
                    assert_eq!(*quant_type, *qt, "QuantGemm quant_type mismatch");
                }
                other => panic!("expected QuantGemm for {:?}, got {:?}", qt, other),
            }
        }
    }

    // @trace TEST-QP-18 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_gather_trace_small_dimensions() {
        // Arrange: minimal valid dimensions (vocab=1, hidden=1)
        let pipeline = QuantPipeline::new(QuantType::Q4_0);

        // Act
        let trace = pipeline.build_gather_trace(1, 1);

        // Assert: trace structure is still valid with minimum dimensions
        assert_eq!(trace.len(), 4);
        match &trace[3] {
            TraceOp::QuantGather { vocab_size, hidden_dim, .. } => {
                assert_eq!(*vocab_size, 1);
                assert_eq!(*hidden_dim, 1);
            }
            other => panic!("expected QuantGather, got {:?}", other),
        }
    }

    // @trace TEST-QP-19 [req:REQ-QPJ] [level:unit]
    #[test]
    fn test_gemm_trace_large_dimensions() {
        // Arrange: large model dimensions
        let pipeline = QuantPipeline::new(QuantType::Q4_0);
        let n = 16384;
        let k = 16384;

        // Act
        let trace = pipeline.build_gemm_trace(0, n, k);

        // Assert: large dimensions should be propagated correctly
        assert_eq!(trace.len(), 4);
        match &trace[3] {
            TraceOp::QuantGemm { m, n: n_val, k: k_val, .. } => {
                assert_eq!(*m, 0);
                assert_eq!(*n_val, 16384);
                assert_eq!(*k_val, 16384);
            }
            other => panic!("expected QuantGemm, got {:?}", other),
        }
    }

    // ── Additional tests ──

    #[test]
    fn test_quant_pipeline_debug_format() {
        let pipeline = QuantPipeline::new(QuantType::Q4_0);
        let debug = format!("{:?}", pipeline);
        assert!(debug.contains("QuantPipeline"));
        assert!(debug.contains("Q4_0"));
        assert!(debug.contains("W256"));
    }

    #[test]
    fn test_quant_pipeline_clone() {
        let pipeline = QuantPipeline::new(QuantType::Q8_0);
        let cloned = pipeline.clone();
        assert_eq!(pipeline.quant_type, cloned.quant_type);
        assert_eq!(pipeline.width, cloned.width);
        assert_eq!(pipeline.compute_dtype, cloned.compute_dtype);
    }

    #[test]
    fn test_quant_layer_traces_debug() {
        let traces = build_quant_layer_traces(
            QuantType::Q4_0, 256, 512, 4, 4, 64,
        );
        let debug = format!("{:?}", traces);
        assert!(debug.contains("QuantLayerTraces"));
    }

    #[test]
    fn test_quant_layer_traces_clone() {
        let traces = build_quant_layer_traces(
            QuantType::Q8_0, 512, 1024, 8, 8, 64,
        );
        let cloned = traces.clone();
        assert_eq!(traces.qkv_dims, cloned.qkv_dims);
        assert_eq!(traces.output_dims, cloned.output_dims);
        assert_eq!(traces.ffn_up_dims, cloned.ffn_up_dims);
        assert_eq!(traces.ffn_down_dims, cloned.ffn_down_dims);
        assert_eq!(traces.ffn_gate_dims, cloned.ffn_gate_dims);
        assert_eq!(traces.quant_type, cloned.quant_type);
    }

    #[test]
    fn test_builder_only_width_override() {
        let pipeline = QuantPipelineBuilder::new(QuantType::Q4_0)
            .width(SimdWidth::W512)
            .build();
        assert_eq!(pipeline.width, SimdWidth::W512);
        // Default compute_dtype should be F32
        assert_eq!(pipeline.compute_dtype, QuantPrecision::F32);
    }

    #[test]
    fn test_builder_only_dtype_override() {
        let pipeline = QuantPipelineBuilder::new(QuantType::Q4_0)
            .compute_dtype(QuantPrecision::BF16)
            .build();
        assert_eq!(pipeline.compute_dtype, QuantPrecision::BF16);
        // Default width should be W256
        assert_eq!(pipeline.width, SimdWidth::W256);
    }

    #[test]
    fn test_pipeline_with_width_fluent_chain() {
        // Verify fluent API returns new Self each time
        let p1 = QuantPipeline::new(QuantType::Q4_0);
        assert_eq!(p1.width, SimdWidth::W256);

        let p2 = p1.with_width(SimdWidth::W128);
        assert_eq!(p2.width, SimdWidth::W128);

        let p3 = p2.with_compute_dtype(QuantPrecision::BF16);
        assert_eq!(p3.compute_dtype, QuantPrecision::BF16);
    }

    #[test]
    fn test_layer_traces_small_model() {
        // Arrange: very small model dimensions
        let traces = build_quant_layer_traces(
            QuantType::Q4_0,
            64,     // hidden
            128,    // intermediate
            2,      // num_heads
            2,      // num_kv_heads
            32,     // head_dim
        );

        // kv_dim = 2*32 = 64, q_dim = 2*32 = 64
        // QKV: N = 64 + 2*64 = 192, K = 64
        assert_eq!(traces.qkv_dims, (0, 192, 64));
        // Output: N = 64, K = 64
        assert_eq!(traces.output_dims, (0, 64, 64));
        // FFN up: N = 128, K = 64
        assert_eq!(traces.ffn_up_dims, (0, 128, 64));
        // FFN down: N = 64, K = 128
        assert_eq!(traces.ffn_down_dims, (0, 64, 128));
        // FFN gate: N = 128, K = 64
        assert_eq!(traces.ffn_gate_dims, (0, 128, 64));
    }

    #[test]
    fn test_gather_trace_fp8_formats() {
        // FP8 formats
        for qt in [QuantType::Fp8E4M3, QuantType::Fp8E5M2] {
            let pipeline = QuantPipeline::new(qt);
            let trace = pipeline.build_gather_trace(32000, 4096);
            assert_eq!(trace.len(), 4);
            assert!(trace.iter().any(|op| matches!(op, TraceOp::QuantGather { .. })),
                "QuantGather missing for {:?}", qt);
        }
    }

    #[test]
    fn test_gemm_trace_nvfp4_and_mxfp4() {
        let formats = [
            QuantType::Nvfp4,
            QuantType::Mxfp4 { block_size: 32 },
        ];
        for qt in &formats {
            let pipeline = QuantPipeline::new(*qt);
            let trace = pipeline.build_gemm_trace(1, 512, 256);
            assert_eq!(trace.len(), 4);

            match &trace[3] {
                TraceOp::QuantGemm { quant_type, m, n, k } => {
                    assert_eq!(*quant_type, *qt);
                    assert_eq!(*m, 1);
                    assert_eq!(*n, 512);
                    assert_eq!(*k, 256);
                }
                other => panic!("expected QuantGemm for {:?}, got {:?}", qt, other),
            }
        }
    }
}
