/// Scalar operator registry: fn_ptr + cached OpTrace per operator.
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

    /// Manually inject an OpTrace (temporary until Phase 0 symexec is implemented).
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
        op_kind: OpKind,
        manual_trace: OpTrace,
    ) {
        // Level 1: structured CFG analysis (handles NormLike, Reduction with multiple loops)
        if let Ok(Some(_)) = self.auto_register_structured(
            key.clone(), sig.clone(), op_kind.clone(),
        ) {
            return;
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
            && self.auto_register_from_symexec(key.clone(), sig.clone(), op_kind).is_ok() {
                return;
            }

        // Level 3: manual trace fallback (or override when Level 2 would lose precision)
        self.register(key.clone(), sig);
        self.inject_trace(key, manual_trace);
    }

    /// Convert an `OpKind` to its hashable `OpKindKey`.
    pub fn key_from_op_kind(kind: &OpKind) -> OpKindKey {
        match kind {
            OpKind::RmsNorm { .. } => OpKindKey::RmsNorm,
            OpKind::LayerNorm { .. } => OpKindKey::LayerNorm,
            OpKind::Gemm { .. } => OpKindKey::Gemm,
            OpKind::GemmBias { .. } => OpKindKey::GemmBias,
            OpKind::Silu => OpKindKey::Silu,
            OpKind::Gelu => OpKindKey::Gelu,
            OpKind::Tanh => OpKindKey::Tanh,
            OpKind::SwiGlu => OpKindKey::SwiGlu,
            OpKind::SwiGluClipped { .. } => OpKindKey::SwiGluClipped,
            OpKind::GeGlu => OpKindKey::GeGlu,
            OpKind::Softmax => OpKindKey::Softmax,
            OpKind::RoPE { .. } => OpKindKey::RoPE,
            OpKind::DualRoPE { .. } => OpKindKey::DualRoPE,
            OpKind::Add => OpKindKey::Add,
            OpKind::Mul => OpKindKey::Mul,
            OpKind::Residual => OpKindKey::Residual,
            OpKind::Transpose { .. } => OpKindKey::Transpose,
            OpKind::Reshape { .. } => OpKindKey::Reshape,
            OpKind::QuantGemm { .. } => OpKindKey::QuantGemm,
            OpKind::Dequantize { .. } => OpKindKey::Dequantize,
            OpKind::MultiHeadAttention { .. } => OpKindKey::MultiHeadAttention,
            OpKind::MeanPool { .. } => OpKindKey::MeanPool,
            OpKind::L2Normalize { .. } => OpKindKey::L2Normalize,
            OpKind::QkNorm { .. } => OpKindKey::QkNorm,
            OpKind::HeadRmsNorm { .. } => OpKindKey::HeadRmsNorm,
            OpKind::ValueNorm { .. } => OpKindKey::ValueNorm,
            OpKind::CachedGQA { .. } => OpKindKey::CachedGQA,
            OpKind::MoEGate { .. } => OpKindKey::MoEGate,
            OpKind::TopK { .. } => OpKindKey::TopK,
            OpKind::WeightedSum { .. } => OpKindKey::WeightedSum,
            OpKind::KvScatterWrite { .. } => OpKindKey::KvScatterWrite,
            // P4/P5 stub variants
            OpKind::VariableLengthBatch => OpKindKey::VariableLengthBatch,
            OpKind::AttentionSkipMask { .. } => OpKindKey::AttentionSkipMask,
            OpKind::FusedRmsNormGemm { .. } => OpKindKey::FusedRmsNormGemm,
            OpKind::ResidualWithTelemetry { .. } => OpKindKey::ResidualWithTelemetry,
            OpKind::EntropyGate { .. } => OpKindKey::EntropyGate,
            OpKind::VRangeQuant { .. } => OpKindKey::VRangeQuant,
            OpKind::KvCentroidPrefetch { .. } => OpKindKey::KvCentroidPrefetch,
            OpKind::LayerBypass { .. } => OpKindKey::LayerBypass,
            OpKind::GateMask { .. } => OpKindKey::GateMask,
            OpKind::MaskedGemm { .. } => OpKindKey::MaskedGemm,
            OpKind::MoEConditionalAdd { .. } => OpKindKey::MoEConditionalAdd,
            OpKind::SoftmaxWithEntropy { .. } => OpKindKey::SoftmaxWithEntropy,
            OpKind::MegaKernelDispatch { .. } => OpKindKey::MegaKernelDispatch,
            OpKind::Gather { .. } => OpKindKey::Gather,
            OpKind::QuantGather { .. } => OpKindKey::QuantGather,
            OpKind::SliceView { .. } => OpKindKey::SliceView,
            OpKind::ColumnSlice { .. } => OpKindKey::ColumnSlice,
            OpKind::AltUpPredict { .. } => OpKindKey::AltUpPredict,
            OpKind::AltUpCorrect { .. } => OpKindKey::AltUpCorrect,
            OpKind::AltUpInject { .. } => OpKindKey::AltUpInject,
            OpKind::DepthwiseConv1D { .. } => OpKindKey::DepthwiseConv1D,
            OpKind::PatchEmbed { .. } => OpKindKey::PatchEmbed,
            OpKind::LearnedPos2D { .. } => OpKindKey::LearnedPos2D,
            OpKind::MoEDispatchPacked { .. } => OpKindKey::MoEDispatchPacked,
            OpKind::MoERouter { .. } => OpKindKey::MoERouter,
            OpKind::QTapSTG { .. } => OpKindKey::QTapSTG,
            OpKind::Argmax { .. } => OpKindKey::Argmax,
            OpKind::StoreToken => OpKindKey::StoreToken,
            OpKind::CheckStopCondition => OpKindKey::CheckStopCondition,
            // Business config ops
            OpKind::WriteLogits { .. } => OpKindKey::WriteLogits,
            OpKind::LogitSoftcap { .. } => OpKindKey::LogitSoftcap,
            OpKind::EarlyExit { .. } => OpKindKey::EarlyExit,
            OpKind::GuardrailCheck { .. } => OpKindKey::GuardrailCheck,
            OpKind::SgInject { .. } => OpKindKey::SgInject,
            OpKind::SgDetect { .. } => OpKindKey::SgDetect,
            OpKind::CotStepCheck { .. } => OpKindKey::CotStepCheck,
            OpKind::SessionKvRestore => OpKindKey::SessionKvRestore,
            OpKind::MmHiddenInject { .. } => OpKindKey::MmHiddenInject,
            OpKind::MtpDraft { .. } => OpKindKey::MtpDraft,
            // MLA
            OpKind::MlaKvCompress { .. } => OpKindKey::MlaKvCompress,
            OpKind::MlaQAbsorb { .. } => OpKindKey::MlaQAbsorb,
            OpKind::MlaVRestore { .. } => OpKindKey::MlaVRestore,
            OpKind::ScaleConst { .. } => OpKindKey::ScaleConst,
            OpKind::MlaAttention { .. } => OpKindKey::MlaAttention,
            OpKind::MlaRopeMerge { .. } => OpKindKey::MlaRopeMerge,
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

}
