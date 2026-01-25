use crate::backend_core::{match_float1, match_float1_mut, BackendCore};
use crate::backend_trait::{TensorSlice, TensorSliceMut};
use crate::ops::sampling::{SamplingConfig, TopKResult};

impl BackendCore {
    pub(crate) fn topk(
        &self,
        logits: TensorSlice<'_>,
        k: usize,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<TopKResult, String> {
        match_float1(
            logits,
            |logits| self.dispatcher.topk::<f32>(logits, k, batch_size, vocab_size),
            |logits| self.dispatcher.topk::<half::f16>(logits, k, batch_size, vocab_size),
            |logits| self.dispatcher.topk::<half::bf16>(logits, k, batch_size, vocab_size),
        )
    }

    pub(crate) fn apply_temperature(
        &self,
        logits: TensorSliceMut<'_>,
        temperature: f32,
    ) -> Result<(), String> {
        match_float1_mut(
            logits,
            |logits| self.dispatcher.apply_temperature::<f32>(logits, temperature),
            |logits| self.dispatcher.apply_temperature::<half::f16>(logits, temperature),
            |logits| self.dispatcher.apply_temperature::<half::bf16>(logits, temperature),
        )
    }

    pub(crate) fn sample_tokens(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<Vec<u32>, String> {
        match_float1(
            logits,
            |logits| self.dispatcher.sample_tokens::<f32>(logits, batch_size, vocab_size, config),
            |logits| self.dispatcher.sample_tokens::<half::f16>(logits, batch_size, vocab_size, config),
            |logits| self.dispatcher.sample_tokens::<half::bf16>(logits, batch_size, vocab_size, config),
        )
    }

    pub(crate) fn argmax(
        &self,
        logits: TensorSlice<'_>,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<Vec<u32>, String> {
        match_float1(
            logits,
            |logits| self.dispatcher.argmax::<f32>(logits, batch_size, vocab_size),
            |logits| self.dispatcher.argmax::<half::f16>(logits, batch_size, vocab_size),
            |logits| self.dispatcher.argmax::<half::bf16>(logits, batch_size, vocab_size),
        )
    }
}
