//! Mamba V3 wrapper with selective scan kernel fallback.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::ops::mamba::{selective_scan_forward_with_kernel, MambaBlock};

/// Configuration for Mamba V3 execution.
#[derive(Debug, Clone, Copy)]
pub struct MambaV3Config {
    /// Enable kernel usage when available.
    pub enable_kernel: bool,
    /// Minimum sequence length before using the kernel.
    pub min_seq_len: usize,
}

impl Default for MambaV3Config {
    fn default() -> Self {
        Self {
            enable_kernel: true,
            min_seq_len: 64,
        }
    }
}

/// Mamba V3 wrapper that prefers selective scan kernels.
#[derive(Debug, Clone)]
pub struct MambaV3<B: Backend> {
    base: MambaBlock<B>,
    config: MambaV3Config,
}

impl<B: Backend> MambaV3<B> {
    /// Create a new Mamba V3 wrapper.
    pub fn new(base: MambaBlock<B>, config: MambaV3Config) -> Self {
        Self { base, config }
    }

    /// Access the underlying Mamba block.
    pub fn base(&self) -> &MambaBlock<B> {
        &self.base
    }

    /// Access the Mamba V3 configuration.
    pub fn config(&self) -> MambaV3Config {
        self.config
    }

    /// Forward selective scan with optional kernel acceleration.
    ///
    /// # Shapes
    /// * `u`: [batch, seq_len, expanded_dim]
    /// * `delta`: [batch, seq_len, expanded_dim]
    /// * `a`: [state_dim, expanded_dim]
    /// * `b`: [batch, seq_len, state_dim]
    /// * `c`: [batch, seq_len, state_dim]
    pub fn forward(
        &self,
        u: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a: Tensor<B, 2>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, &'static str> {
        let expanded_dim = u.dims()[2];
        let expected_expanded = self.base.input_dim() * self.base.expand_ratio();
        if expanded_dim != expected_expanded {
            return Err("u expanded_dim mismatch for MambaV3");
        }

        let seq_len = u.dims()[1];
        let use_kernel = self.should_use_kernel(seq_len);
        selective_scan_forward_with_kernel(u, delta, a, b, c, use_kernel)
    }

    fn should_use_kernel(&self, seq_len: usize) -> bool {
        self.config.enable_kernel && seq_len >= self.config.min_seq_len && cfg!(feature = "mamba-kernel")
    }
}
