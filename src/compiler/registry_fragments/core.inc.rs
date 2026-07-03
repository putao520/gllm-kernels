/// Scalar operator registry: fn_ptr + cached OpTrace per operator.
// @trace REQ-AIS-007 [entity:ENT-AUTO-INSTR-SELECT] [api:POST /compile/scalar-registry]
pub struct ScalarOpRegistry {
    entries: HashMap<OpKindKey, ScalarFnSignature>,
    trace_cache: HashMap<OpKindKey, OpTrace>,
}

impl ScalarOpRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        ScalarOpRegistry {
            entries: HashMap::new(),
            trace_cache: HashMap::new(),
        }
    }

    /// Register a scalar function signature.
    pub fn register(&mut self, key: OpKindKey, sig: ScalarFnSignature) {
        self.entries.insert(key, sig);
    }

    /// Get the cached OpTrace for a key (returns `None` if not yet injected).
    pub fn get_trace(&self, key: &OpKindKey) -> Option<&OpTrace> {
        self.trace_cache.get(key)
    }

    /// Get the scalar function signature for a key.
    pub fn get_signature(&self, key: &OpKindKey) -> Option<&ScalarFnSignature> {
        self.entries.get(key)
    }

    /// Manually inject an OpTrace (temporary until Scalar + SymExec symexec is implemented).
    pub fn inject_trace(&mut self, key: OpKindKey, trace: OpTrace) {
        self.trace_cache.insert(key, trace);
    }

    /// Three-level fallback: structured CFG → linear symexec → manual trace.
    ///
    /// 1. `auto_register_structured` — CFG multi-loop analysis (most precise).
    /// 2. `auto_register_from_symexec` — linear symbolic execution.
    /// 3. Manual trace injection — last resort, BUT overrides Level 2 when
    ///    Level 2's `classify_pattern` loses multi-output precision.
    fn register_with_symexec_fallback(
        &mut self,
        key: OpKindKey,
        sig: ScalarFnSignature,
        manual_trace: OpTrace,
    ) {
        // Level 1: structured CFG analysis (handles NormLike, Reduction with multiple loops)
        //
        // BCE-20260704-STRUCTURED-SYMEXEC-LOOP-MISCLASSIFY: structured CFG analysis
        // can misclassify a 2-logical-loop NormLike (e.g. RmsNorm, which has only
        // `x` + `weight` ptr params and no bias) as a 3-loop LayerNorm pattern whose
        // transform references `Input(3)` (weight) + `Input(4)` (bias). When the
        // caller later emits the op as RmsNorm it only supplies 3 input VRegs, so
        // `Input(3)` is out of bounds → panic.
        //
        // Defense: verify the generated pattern's transform/reduce/finalize
        // `Input(n)` arity does not exceed the function signature's pointer-parameter
        // count. If the pattern references Input slots the function does not have,
        // discard the structured result and fall through to Level 2/3 (manual trace).
        if let Ok(Some(pattern)) = self.auto_register_structured(
            key.clone(), sig.clone(),
        ) {
            let max_input = Self::max_input_arity(&pattern);
            let n_ptr_params = sig.params.iter()
                .filter(|p| matches!(p, ScalarParam::InputPtr | ScalarParam::WeightPtr | ScalarParam::OutputPtr))
                .count();
            if max_input <= n_ptr_params {
                return; // Level 1 succeeded with a pattern consistent with the signature.
            }
            // Pattern references Input(n) beyond what the signature provides →
            // structured analysis misclassified (e.g. RmsNorm → LayerNorm). Fall
            // through to Level 2/3 so the correct manual trace is used.
        }

        // Determine if manual trace has precision that Level 2 symexec may lose:
        // 1. Multi-output: Level 2 hardcodes num_outputs: 1
        // 2. BinaryElementwise: Level 2 symexec may confuse Input(0)/Input(1) in release
        //    mode due to compiler optimizations (SROA/inlining of scalar fn pointers)
        let manual_num_outputs = match &manual_trace.pattern {
            ComputePattern::Injective { num_outputs, .. } => *num_outputs,
            _ => 1,
        };
        let is_binary = matches!(&manual_trace.pattern, ComputePattern::BinaryElementwise { .. });

        // Level 2: linear symexec — skip when manual trace has higher precision
        if manual_num_outputs <= 1 && !is_binary
            && self.auto_register_from_symexec(key.clone(), sig.clone()).is_ok() {
                return;
            }

        // Level 3: manual trace fallback (or override when Level 2 would lose precision)
        self.register(key.clone(), sig);
        self.inject_trace(key, manual_trace);
    }

    /// Convert an `Op` to its hashable `OpKindKey` (OpKind enum 已删除，从 Op 派生)。
    ///
    /// OpKindKey 作为 ScalarOpRegistry 的 HashMap key（Op 本身因含 f32/f64 字段无法 Hash）。
    /// 对于 norm/gemm 等 op，key 是参数无关的（eps/trans_b 等参数不影响 trace 模板）。
    pub fn key_from_op(op: &crate::compiler::graph::Op) -> OpKindKey {
        use crate::compiler::graph::Op;
        match op {
            Op::RmsNorm(_) => OpKindKey::RmsNorm,
            Op::LayerNorm(_) => OpKindKey::LayerNorm,
            Op::Gemm(_) => OpKindKey::Gemm,
            Op::GemmBias(_) => OpKindKey::GemmBias,
            Op::Silu => OpKindKey::Silu,
            Op::Gelu => OpKindKey::Gelu,
            Op::Tanh => OpKindKey::Tanh,
            Op::Sigmoid => OpKindKey::Sigmoid,
            Op::SwiGlu => OpKindKey::SwiGlu,
            Op::SwiGluClipped { .. } => OpKindKey::SwiGluClipped,
            Op::GeGlu => OpKindKey::GeGlu,
            Op::Softmax => OpKindKey::Softmax,
            Op::RoPE(_) => OpKindKey::RoPE,
            Op::DualRoPE(_) => OpKindKey::DualRoPE,
            Op::Add => OpKindKey::Add,
            Op::Mul => OpKindKey::Mul,
            Op::Sub => OpKindKey::Sub,
            Op::Div => OpKindKey::Div,
            Op::Pow => OpKindKey::Pow,
            Op::Sqrt => OpKindKey::Sqrt,
            Op::Residual => OpKindKey::Residual,
            Op::Transpose { .. } => OpKindKey::Transpose,
            Op::Reshape { .. } => OpKindKey::Reshape,
            Op::QuantGemm(_) => OpKindKey::QuantGemm,
            Op::Dequantize { .. } => OpKindKey::Dequantize,
            Op::MultiHeadAttention(_) => OpKindKey::MultiHeadAttention,
            Op::MeanPool { .. } => OpKindKey::MeanPool,
            Op::L2Normalize { .. } => OpKindKey::L2Normalize,
            Op::QkNorm { .. } => OpKindKey::QkNorm,
            Op::HeadRmsNorm { .. } => OpKindKey::HeadRmsNorm,
            Op::ValueNorm(_) => OpKindKey::ValueNorm,
            Op::CachedGqa(_) => OpKindKey::CachedGQA,
            Op::MoEGate { .. } => OpKindKey::MoEGate,
            Op::TopK { .. } => OpKindKey::TopK,
            Op::WeightedSum { .. } => OpKindKey::WeightedSum,
            Op::KvScatterWrite { .. } => OpKindKey::KvScatterWrite,
            // @trace REQ-FATOP-025: KvCacheWrite 已物理删除
            // P4/P5 stub variants
            Op::VariableLengthBatch => OpKindKey::VariableLengthBatch,
            Op::AttentionSkipMask { .. } => OpKindKey::AttentionSkipMask,
            Op::FusedRmsNormGemm { .. } => OpKindKey::FusedRmsNormGemm,
            Op::ResidualWithTelemetry { .. } => OpKindKey::ResidualWithTelemetry,
            Op::EntropyGate { .. } => OpKindKey::EntropyGate,
            Op::VRangeQuant { .. } => OpKindKey::VRangeQuant,
            Op::KvCentroidPrefetch { .. } => OpKindKey::KvCentroidPrefetch,
            Op::LayerBypass { .. } => OpKindKey::LayerBypass,
            Op::GateMask { .. } => OpKindKey::GateMask,
            Op::MaskedGemm { .. } => OpKindKey::MaskedGemm,
            Op::MoEConditionalAdd { .. } => OpKindKey::MoEConditionalAdd,
            Op::SoftmaxWithEntropy { .. } => OpKindKey::SoftmaxWithEntropy,
            Op::MegaKernelDispatch { .. } => OpKindKey::MegaKernelDispatch,
            Op::Gather { .. } => OpKindKey::Gather,
            Op::QuantGather { .. } => OpKindKey::QuantGather,
            Op::SliceView { .. } => OpKindKey::SliceView,
            Op::ColumnSlice { .. } => OpKindKey::ColumnSlice,
            Op::AltUpPredict { .. } => OpKindKey::AltUpPredict,
            Op::AltUpCorrect { .. } => OpKindKey::AltUpCorrect,
            Op::AltUpInject { .. } => OpKindKey::AltUpInject,
            Op::DepthwiseConv1D { .. } => OpKindKey::DepthwiseConv1D,
            Op::PatchEmbed { .. } => OpKindKey::PatchEmbed,
            Op::LearnedPos2D { .. } => OpKindKey::LearnedPos2D,
            Op::MoEDispatchPacked { .. } => OpKindKey::MoEDispatchPacked,
            Op::MoERouter { .. } => OpKindKey::MoERouter,
            Op::QTapSTG { .. } => OpKindKey::QTapSTG,
            Op::Argmax { .. } => OpKindKey::Argmax,
            Op::StoreToken => OpKindKey::StoreToken,
            Op::CheckStopCondition => OpKindKey::CheckStopCondition,
            // Business config ops
            Op::WriteLogits { .. } => OpKindKey::WriteLogits,
            Op::LogitSoftcap { .. } => OpKindKey::LogitSoftcap,
            Op::EarlyExit { .. } => OpKindKey::EarlyExit,
            Op::GuardrailCheck { .. } => OpKindKey::GuardrailCheck,
            Op::SgInject { .. } => OpKindKey::SgInject,
            Op::SgDetect { .. } => OpKindKey::SgDetect,
            Op::CotStepCheck { .. } => OpKindKey::CotStepCheck,
            Op::SessionKvRestore => OpKindKey::SessionKvRestore,
            Op::MmHiddenInject { .. } => OpKindKey::MmHiddenInject,
            Op::MtpDraft { .. } => OpKindKey::MtpDraft,
            // MLA
            Op::MlaKvCompress { .. } => OpKindKey::MlaKvCompress,
            Op::MlaQAbsorb { .. } => OpKindKey::MlaQAbsorb,
            Op::MlaVRestore { .. } => OpKindKey::MlaVRestore,
            Op::ScaleConst { .. } => OpKindKey::ScaleConst,
            Op::MlaAttention(_) => OpKindKey::MlaAttention,
            Op::MlaRopeMerge { .. } => OpKindKey::MlaRopeMerge,
            Op::Relu => OpKindKey::Relu,
            Op::Exp => OpKindKey::Exp,
            Op::Erf => OpKindKey::Erf,
        }
    }

    /// Number of registered scalar functions.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Number of cached OpTraces.
    pub fn num_traces(&self) -> usize {
        self.trace_cache.len()
    }

    /// Return all registered OpKindKeys.
    ///
    /// Used by BackendCapMatrix::build() to derive the capability matrix
    /// from the registry's registration coverage.
    // @trace REQ-BACKEND-CAP-002 [entity:ENT-BACKEND-CAP-MATRIX]
    pub fn registered_keys(&self) -> Vec<OpKindKey> {
        self.entries.keys().cloned().collect()
    }

    /// Maximum `TraceOp::Input(n)` index referenced across every TraceOp slice
    /// embedded in `pattern`, plus one (i.e. the Input arity the pattern demands).
    ///
    /// Returns `0` when the pattern references no `TraceOp::Input` at all.
    ///
    /// BCE-20260704-STRUCTURED-SYMEXEC-LOOP-MISCLASSIFY: used by
    /// `register_with_symexec_fallback` to verify a structured-analysis pattern
    /// does not reference Input slots the scalar function signature lacks
    /// (e.g. an RmsNorm pattern misclassified as LayerNorm referencing
    /// `Input(3)`/`Input(4)` for weight/bias that RmsNorm does not have).
    fn max_input_arity(pattern: &ComputePattern) -> usize {
        let mut max: Option<u32> = None;
        let mut consider = |ops: &[TraceOp]| {
            for op in ops {
                if let TraceOp::Input(n) = op {
                    if *n + 1 > max.unwrap_or(0) {
                        max = Some(*n + 1);
                    }
                }
            }
        };

        match pattern {
            ComputePattern::Elementwise { body }
            | ComputePattern::BinaryElementwise { body } => consider(body),
            ComputePattern::Injective { body, .. } => consider(body),
            ComputePattern::Reduction { combine, second_pass, normalize, .. } => {
                consider(combine);
                if let Some(sp) = second_pass {
                    consider(&sp.element_transform);
                    consider(&sp.combine);
                }
                if let Some(norm) = normalize {
                    consider(norm);
                }
            }
            ComputePattern::NormLike { reduce, finalize, transform } => {
                consider(reduce);
                consider(finalize);
                consider(transform);
            }
            ComputePattern::QuantDecode { decode, .. } => consider(decode),
            ComputePattern::Gemm => {}
        }
        max.unwrap_or(0) as usize
    }

}
