impl ScalarOpRegistry {

    /// Auto-register a scalar function by running symbolic execution to extract its trace.
    ///
    /// On x86_64 with the `jit-x86` feature, this disassembles the compiled scalar
    /// function via iced-x86 and feeds each instruction into the symbolic execution
    /// engine to extract the computational structure automatically.
    ///
    /// If the key is already registered, the existing entry is overwritten
    /// (allows symexec to upgrade a manual trace).
    pub fn auto_register_from_symexec(
        &mut self,
        key: OpKindKey,
        fn_sig: ScalarFnSignature,
        op_kind: OpKind,
    ) -> Result<ComputePattern, RegistryError> {
        let trace_ops = self.run_symexec(&fn_sig)?;

        if trace_ops.is_empty() {
            return Err(RegistryError::EmptyTrace);
        }

        // Reject trivial traces (e.g. just [Input(0)]) — these indicate the
        // symexec engine didn't actually analyze the function body.
        let has_compute = trace_ops.iter().any(|op| !matches!(op, TraceOp::Input(_) | TraceOp::Const(_)));
        if !has_compute {
            return Err(RegistryError::EmptyTrace);
        }

        let pattern = classify_pattern(&trace_ops);
        let trace = OpTrace {
            op_kind,
            pattern: pattern.clone(),
            signature: fn_sig.clone(),
        };
        self.entries.insert(key.clone(), fn_sig);
        self.trace_cache.insert(key, trace);
        Ok(pattern)
    }

    /// Run symbolic execution on a scalar function to extract its trace ops.
    ///
    /// Uses the decoder bridge (iced-x86 disassembly → symexec) when available,
    /// otherwise falls back to the stub path (no instructions → empty trace).
    fn run_symexec(&self, fn_sig: &ScalarFnSignature) -> Result<Vec<TraceOp>, RegistryError> {
        // Decoder bridge: disassemble the compiled function and symbolically execute it.
        #[cfg(feature = "jit-x86")]
        {
            use crate::compiler::symexec::decoder::analyze_scalar_fn;
            unsafe { analyze_scalar_fn(fn_sig.fn_ptr, fn_sig) }
                .map_err(RegistryError::SymExec)
        }

        // Fallback: no decoder available, create an empty executor.
        #[cfg(not(feature = "jit-x86"))]
        {
            let n_float = fn_sig.params.iter().filter(|p| matches!(p, ScalarParam::Scalar(_))).count();
            let n_ptr = fn_sig.params.iter().filter(|p| matches!(p, ScalarParam::InputPtr | ScalarParam::OutputPtr | ScalarParam::WeightPtr)).count();
            let executor = SymbolicExecutor::new(n_float, n_ptr);
            executor.extract_trace().map_err(RegistryError::SymExec)
        }
    }

    /// Auto-register a multi-loop scalar function using structured CFG analysis.
    ///
    /// This is the Phase 3 upgrade path: instead of linear symexec (which only
    /// handles single-loop elementwise ops), this uses CFG → loop detection →
    /// multi-pass combination to classify NormLike and Reduction patterns.
    ///
    /// Returns `Ok(Some(pattern))` if structured analysis succeeded,
    /// `Ok(None)` if the function has no loops (caller should try linear symexec),
    /// or `Err` on failure.
    pub fn auto_register_structured(
        &mut self,
        key: OpKindKey,
        fn_sig: ScalarFnSignature,
        op_kind: OpKind,
    ) -> Result<Option<ComputePattern>, RegistryError> {
        #[cfg(feature = "jit-x86")]
        {
            use crate::compiler::symexec::decoder::analyze_scalar_fn_structured;

            match unsafe { analyze_scalar_fn_structured(fn_sig.fn_ptr, &fn_sig) }
                .map_err(RegistryError::SymExec)?
            {
                Some(analysis) => {
                    let pattern = analysis.pattern.clone();
                    let trace = OpTrace {
                        op_kind,
                        pattern: pattern.clone(),
                        signature: fn_sig.clone(),
                    };
                    self.entries.insert(key.clone(), fn_sig);
                    self.trace_cache.insert(key, trace);
                    Ok(Some(pattern))
                }
                None => Ok(None), // No loops → fall back to linear.
            }
        }

        #[cfg(not(feature = "jit-x86"))]
        {
            let _ = (key, fn_sig, op_kind);
            Ok(None)
        }
    }
}

