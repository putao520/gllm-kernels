use crate::backend_core::{
    match_float1_mut, match_float1_out, match_float2_out, match_float2_out2, match_float3_out,
    BackendCore,
};
use crate::backend_trait::{TensorSlice, TensorSliceMut};
use crate::kernel_dispatcher::{
    FlashAttentionConfig, MatmulConfig, PagedAttentionConfig, SoftmaxConfig,
};
use crate::ops::rope::RoPEConfig;

impl BackendCore {
    pub(crate) fn flash_attention(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        v: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: FlashAttentionConfig,
    ) -> Result<(), String> {
        match_float3_out(
            "flash_attention",
            q,
            k,
            v,
            output,
            |q, k, v, out| self.dispatcher.flash_attention::<f32>(q, k, v, out, config.clone()),
            |q, k, v, out| self.dispatcher.flash_attention::<half::f16>(q, k, v, out, config.clone()),
            |q, k, v, out| self.dispatcher.flash_attention::<half::bf16>(q, k, v, out, config.clone()),
        )
    }

    pub(crate) fn paged_attention(
        &self,
        q: TensorSlice<'_>,
        k_cache: TensorSlice<'_>,
        v_cache: TensorSlice<'_>,
        page_table: &[u32],
        seq_lens: &[u32],
        output: TensorSliceMut<'_>,
        config: PagedAttentionConfig,
    ) -> Result<(), String> {
        match_float3_out(
            "paged_attention",
            q,
            k_cache,
            v_cache,
            output,
            |q, k, v, out| {
                self.dispatcher
                    .paged_attention::<f32>(q, k, v, page_table, seq_lens, out, config.clone())
            },
            |q, k, v, out| {
                self.dispatcher
                    .paged_attention::<half::f16>(q, k, v, page_table, seq_lens, out, config.clone())
            },
            |q, k, v, out| {
                self.dispatcher
                    .paged_attention::<half::bf16>(q, k, v, page_table, seq_lens, out, config.clone())
            },
        )
    }

    pub(crate) fn softmax(
        &self,
        input: TensorSlice<'_>,
        output: TensorSliceMut<'_>,
        config: SoftmaxConfig,
    ) -> Result<(), String> {
        match_float1_out(
            "softmax",
            input,
            output,
            |input, out| self.dispatcher.softmax::<f32>(input, out, config.clone()),
            |input, out| self.dispatcher.softmax::<half::f16>(input, out, config.clone()),
            |input, out| self.dispatcher.softmax::<half::bf16>(input, out, config.clone()),
        )
    }

    pub(crate) fn matmul(
        &self,
        a: TensorSlice<'_>,
        b: TensorSlice<'_>,
        c: TensorSliceMut<'_>,
        config: MatmulConfig,
    ) -> Result<(), String> {
        match_float2_out(
            "matmul",
            a,
            b,
            c,
            |a, b, c| self.dispatcher.matmul::<f32>(a, b, c, config.clone()),
            |a, b, c| self.dispatcher.matmul::<half::f16>(a, b, c, config.clone()),
            |a, b, c| self.dispatcher.matmul::<half::bf16>(a, b, c, config.clone()),
        )
    }

    pub(crate) fn rope_precompute(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
        config: RoPEConfig,
    ) -> Result<(), String> {
        self.dispatcher.rope_precompute(cos_out, sin_out, config);
        Ok(())
    }

    pub(crate) fn rope_apply(
        &self,
        q: TensorSlice<'_>,
        k: TensorSlice<'_>,
        cos_cache: &[f32],
        sin_cache: &[f32],
        q_out: TensorSliceMut<'_>,
        k_out: TensorSliceMut<'_>,
        batch_size: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), String> {
        match_float2_out2(
            "rope_apply",
            q,
            k,
            q_out,
            k_out,
            |q, k, q_out, k_out| {
                self.dispatcher.rope_apply::<f32>(
                    q,
                    k,
                    cos_cache,
                    sin_cache,
                    q_out,
                    k_out,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    position_offset,
                );
            },
            |q, k, q_out, k_out| {
                self.dispatcher.rope_apply::<half::f16>(
                    q,
                    k,
                    cos_cache,
                    sin_cache,
                    q_out,
                    k_out,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    position_offset,
                );
            },
            |q, k, q_out, k_out| {
                self.dispatcher.rope_apply::<half::bf16>(
                    q,
                    k,
                    cos_cache,
                    sin_cache,
                    q_out,
                    k_out,
                    batch_size,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    position_offset,
                );
            },
        )
    }

    pub(crate) fn rope_apply_inplace(
        &self,
        x: TensorSliceMut<'_>,
        cos_cache: &[f32],
        sin_cache: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> Result<(), String> {
        match_float1_mut(
            x,
            |x| {
                self.dispatcher.rope_apply_inplace::<f32>(
                    x,
                    cos_cache,
                    sin_cache,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    position_offset,
                );
            },
            |x| {
                self.dispatcher.rope_apply_inplace::<half::f16>(
                    x,
                    cos_cache,
                    sin_cache,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    position_offset,
                );
            },
            |x| {
                self.dispatcher.rope_apply_inplace::<half::bf16>(
                    x,
                    cos_cache,
                    sin_cache,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    position_offset,
                );
            },
        )
    }
}
