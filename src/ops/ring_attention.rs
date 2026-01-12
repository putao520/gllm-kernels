//! Ring Attention for distributed long-sequence processing.

use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::comm::{Communicator, TensorMessage};
use crate::ops::flash_attention::HierarchicalFlashAttention;
use crate::types::AttentionConfig;

/// Ring Attention configuration.
#[derive(Clone, Debug)]
pub struct RingAttentionConfig {
    /// Number of devices/nodes participating in the ring.
    pub world_size: usize,
    /// Rank of the current device/node.
    pub rank: usize,
    /// Local KV sequence length on each device.
    pub local_seq_len: usize,
    /// Whether to apply causal masking.
    pub causal: bool,
    /// Communication backend for KV transfer.
    pub comm_backend: CommBackend,
}

/// Communication backend.
#[derive(Clone, Debug, Default)]
pub enum CommBackend {
    /// Single node, shared memory.
    #[default]
    SharedMemory,
    /// Multi-node, NCCL.
    Nccl,
    /// Multi-node, TCP.
    Tcp,
}

/// Ring Attention implementation.
pub struct RingAttention {
    config: RingAttentionConfig,
    local_attention: HierarchicalFlashAttention,
}

impl RingAttention {
    pub fn new(config: RingAttentionConfig) -> Self {
        let local_attention = HierarchicalFlashAttention::default_config();
        Self {
            config,
            local_attention,
        }
    }

    /// Ring Attention forward pass.
    pub fn forward<B: Backend>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        config: &AttentionConfig,
    ) -> Tensor<B, 4> {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();
        let key_len = k.dims()[2];

        if query_len == 0 || key_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let world_size = self.config.world_size.max(1);
        let rank = self.config.rank % world_size;
        let local_seq_len = self.config.local_seq_len.max(1);
        let q_offset = rank * local_seq_len;
        let causal = self.config.causal;

        let block_q = config.block_q.max(1);
        let block_kv = config.block_kv.max(1);
        let scale = if config.scale > 0.0 {
            config.scale
        } else {
            1.0 / (head_dim as f32).sqrt()
        };
        let use_log_space = self.local_attention.config().use_log_space;

        let num_q_blocks = (query_len + block_q - 1) / block_q;
        let mut outputs = Vec::with_capacity(num_q_blocks);
        let mut q_start = 0usize;

        while q_start < query_len {
            let q_end = (q_start + block_q).min(query_len);
            let q_block_len = q_end - q_start;
            let q_block = q.clone().slice([
                0..batch_size,
                0..num_heads,
                q_start..q_end,
                0..head_dim,
            ]);

            if use_log_space {
                let mut m_i = Tensor::<B, 4>::full(
                    [batch_size, num_heads, q_block_len, 1],
                    f32::NEG_INFINITY,
                    &device,
                );
                let mut log_l_i = Tensor::<B, 4>::full(
                    [batch_size, num_heads, q_block_len, 1],
                    f32::NEG_INFINITY,
                    &device,
                );
                let mut o_i = Tensor::<B, 4>::zeros(
                    [batch_size, num_heads, q_block_len, head_dim],
                    &device,
                );

                let mut current_k = k.clone();
                let mut current_v = v.clone();

                for step in 0..world_size {
                    let kv_rank = (rank + step) % world_size;
                    let need_compute = if causal { kv_rank <= rank } else { true };

                    if need_compute {
                        let shard_offset = kv_rank * local_seq_len;
                        let shard_len = current_k.dims()[2];
                        let mut kv_start = 0usize;

                        while kv_start < shard_len {
                            let kv_end = (kv_start + block_kv).min(shard_len);
                            let kv_block_len = kv_end - kv_start;

                            let k_block = current_k.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);
                            let v_block = current_v.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);

                            let mut scores = q_block.clone().matmul(k_block.transpose()) * scale;

                            if causal {
                                let mask = Self::build_causal_mask::<B>(
                                    &device,
                                    q_block_len,
                                    kv_block_len,
                                    q_start,
                                    kv_start,
                                    q_offset,
                                    shard_offset,
                                );
                                scores = scores + mask;
                            }

                            let m_ij = scores.clone().max_dim(3);
                            let m_new = m_i.clone().max_pair(m_ij.clone());

                            let scores_shifted = scores - m_ij.clone();
                            let p_ij = scores_shifted.exp();
                            let sum_p = p_ij.clone().sum_dim(3);
                            let log_sum_p = sum_p.log();

                            let m_diff = m_i - m_new.clone();
                            let log_prev = m_diff.clone() + log_l_i;
                            let log_curr = (m_ij - m_new.clone()) + log_sum_p;
                            let log_l_new = Self::tensor_log_add_exp(log_prev, log_curr);

                            let m_scale = m_diff.exp();
                            o_i = m_scale * o_i + p_ij.matmul(v_block);

                            m_i = m_new;
                            log_l_i = log_l_new;
                            kv_start = kv_end;
                        }
                    }

                    if step < world_size - 1 {
                        (current_k, current_v) = self.ring_send_recv(current_k, current_v);
                    }
                }

                let l_i = log_l_i.exp();
                outputs.push(o_i / l_i);
            } else {
                let mut m_i = Tensor::<B, 4>::full(
                    [batch_size, num_heads, q_block_len, 1],
                    f32::NEG_INFINITY,
                    &device,
                );
                let mut l_i =
                    Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
                let mut o_i = Tensor::<B, 4>::zeros(
                    [batch_size, num_heads, q_block_len, head_dim],
                    &device,
                );

                let mut current_k = k.clone();
                let mut current_v = v.clone();

                for step in 0..world_size {
                    let kv_rank = (rank + step) % world_size;
                    let need_compute = if causal { kv_rank <= rank } else { true };

                    if need_compute {
                        let shard_offset = kv_rank * local_seq_len;
                        let shard_len = current_k.dims()[2];
                        let mut kv_start = 0usize;

                        while kv_start < shard_len {
                            let kv_end = (kv_start + block_kv).min(shard_len);
                            let kv_block_len = kv_end - kv_start;

                            let k_block = current_k.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);
                            let v_block = current_v.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);

                            let mut scores = q_block.clone().matmul(k_block.transpose()) * scale;

                            if causal {
                                let mask = Self::build_causal_mask::<B>(
                                    &device,
                                    q_block_len,
                                    kv_block_len,
                                    q_start,
                                    kv_start,
                                    q_offset,
                                    shard_offset,
                                );
                                scores = scores + mask;
                            }

                            let m_ij = scores.clone().max_dim(3);
                            let m_new = m_i.clone().max_pair(m_ij);

                            let m_scale = (m_i - m_new.clone()).exp();
                            let p_ij = (scores - m_new.clone()).exp();
                            let p_sum = p_ij.clone().sum_dim(3);

                            l_i = m_scale.clone() * l_i + p_sum;
                            o_i = m_scale * o_i + p_ij.matmul(v_block);
                            m_i = m_new;

                            kv_start = kv_end;
                        }
                    }

                    if step < world_size - 1 {
                        (current_k, current_v) = self.ring_send_recv(current_k, current_v);
                    }
                }

                outputs.push(o_i / l_i);
            }

            q_start = q_end;
        }

        Tensor::cat(outputs, 2)
    }

    fn ring_send_recv<B: Backend>(
        &self,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Placeholder: actual transport integration depends on the comm backend.
        match self.config.comm_backend {
            CommBackend::SharedMemory | CommBackend::Nccl | CommBackend::Tcp => (k, v),
        }
    }

    /// Ring Attention forward pass with real distributed communication.
    ///
    /// This method uses a `Communicator` to perform actual KV tensor exchange
    /// between ranks in the ring.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, heads, seq_len, head_dim]
    /// * `k` - Local key tensor [batch, heads, local_seq_len, head_dim]
    /// * `v` - Local value tensor [batch, heads, local_seq_len, head_dim]
    /// * `config` - Attention configuration
    /// * `comm` - Communicator for ring communication
    pub fn forward_distributed<B, C>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        config: &AttentionConfig,
        comm: Arc<C>,
    ) -> Tensor<B, 4>
    where
        B: Backend,
        C: Communicator<Data = TensorMessage> + 'static,
    {
        let device = q.device();
        let [batch_size, num_heads, query_len, head_dim] = q.dims();
        let key_len = k.dims()[2];

        if query_len == 0 || key_len == 0 {
            return Tensor::zeros([batch_size, num_heads, query_len, head_dim], &device);
        }

        let world_size = comm.world_size().max(1);
        let rank = comm.rank() % world_size;
        let local_seq_len = self.config.local_seq_len.max(1);
        let q_offset = rank * local_seq_len;
        let causal = self.config.causal;

        let block_q = config.block_q.max(1);
        let block_kv = config.block_kv.max(1);
        let scale = if config.scale > 0.0 {
            config.scale
        } else {
            1.0 / (head_dim as f32).sqrt()
        };
        let use_log_space = self.local_attention.config().use_log_space;

        let num_q_blocks = (query_len + block_q - 1) / block_q;
        let mut outputs = Vec::with_capacity(num_q_blocks);
        let mut q_start = 0usize;

        while q_start < query_len {
            let q_end = (q_start + block_q).min(query_len);
            let q_block_len = q_end - q_start;
            let q_block = q.clone().slice([
                0..batch_size,
                0..num_heads,
                q_start..q_end,
                0..head_dim,
            ]);

            if use_log_space {
                let mut m_i = Tensor::<B, 4>::full(
                    [batch_size, num_heads, q_block_len, 1],
                    f32::NEG_INFINITY,
                    &device,
                );
                let mut log_l_i = Tensor::<B, 4>::full(
                    [batch_size, num_heads, q_block_len, 1],
                    f32::NEG_INFINITY,
                    &device,
                );
                let mut o_i = Tensor::<B, 4>::zeros(
                    [batch_size, num_heads, q_block_len, head_dim],
                    &device,
                );

                let mut current_k = k.clone();
                let mut current_v = v.clone();

                for step in 0..world_size {
                    let kv_rank = (rank + step) % world_size;
                    let need_compute = if causal { kv_rank <= rank } else { true };

                    if need_compute {
                        let shard_offset = kv_rank * local_seq_len;
                        let shard_len = current_k.dims()[2];
                        let mut kv_start = 0usize;

                        while kv_start < shard_len {
                            let kv_end = (kv_start + block_kv).min(shard_len);
                            let kv_block_len = kv_end - kv_start;

                            let k_block = current_k.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);
                            let v_block = current_v.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);

                            let mut scores = q_block.clone().matmul(k_block.transpose()) * scale;

                            if causal {
                                let mask = Self::build_causal_mask::<B>(
                                    &device,
                                    q_block_len,
                                    kv_block_len,
                                    q_start,
                                    kv_start,
                                    q_offset,
                                    shard_offset,
                                );
                                scores = scores + mask;
                            }

                            let m_ij = scores.clone().max_dim(3);
                            let m_new = m_i.clone().max_pair(m_ij.clone());

                            let scores_shifted = scores - m_ij.clone();
                            let p_ij = scores_shifted.exp();
                            let sum_p = p_ij.clone().sum_dim(3);
                            let log_sum_p = sum_p.log();

                            let m_diff = m_i - m_new.clone();
                            let log_prev = m_diff.clone() + log_l_i;
                            let log_curr = (m_ij - m_new.clone()) + log_sum_p;
                            let log_l_new = Self::tensor_log_add_exp(log_prev, log_curr);

                            let m_scale = m_diff.exp();
                            o_i = m_scale * o_i + p_ij.matmul(v_block);

                            m_i = m_new;
                            log_l_i = log_l_new;
                            kv_start = kv_end;
                        }
                    }

                    if step < world_size - 1 {
                        (current_k, current_v) =
                            Self::ring_send_recv_with_comm(&current_k, &current_v, &*comm, &device);
                    }
                }

                let l_i = log_l_i.exp();
                outputs.push(o_i / l_i);
            } else {
                let mut m_i = Tensor::<B, 4>::full(
                    [batch_size, num_heads, q_block_len, 1],
                    f32::NEG_INFINITY,
                    &device,
                );
                let mut l_i =
                    Tensor::<B, 4>::zeros([batch_size, num_heads, q_block_len, 1], &device);
                let mut o_i = Tensor::<B, 4>::zeros(
                    [batch_size, num_heads, q_block_len, head_dim],
                    &device,
                );

                let mut current_k = k.clone();
                let mut current_v = v.clone();

                for step in 0..world_size {
                    let kv_rank = (rank + step) % world_size;
                    let need_compute = if causal { kv_rank <= rank } else { true };

                    if need_compute {
                        let shard_offset = kv_rank * local_seq_len;
                        let shard_len = current_k.dims()[2];
                        let mut kv_start = 0usize;

                        while kv_start < shard_len {
                            let kv_end = (kv_start + block_kv).min(shard_len);
                            let kv_block_len = kv_end - kv_start;

                            let k_block = current_k.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);
                            let v_block = current_v.clone().slice([
                                0..batch_size,
                                0..num_heads,
                                kv_start..kv_end,
                                0..head_dim,
                            ]);

                            let mut scores = q_block.clone().matmul(k_block.transpose()) * scale;

                            if causal {
                                let mask = Self::build_causal_mask::<B>(
                                    &device,
                                    q_block_len,
                                    kv_block_len,
                                    q_start,
                                    kv_start,
                                    q_offset,
                                    shard_offset,
                                );
                                scores = scores + mask;
                            }

                            let m_ij = scores.clone().max_dim(3);
                            let m_new = m_i.clone().max_pair(m_ij);

                            let m_scale = (m_i - m_new.clone()).exp();
                            let p_ij = (scores - m_new.clone()).exp();
                            let p_sum = p_ij.clone().sum_dim(3);

                            l_i = m_scale.clone() * l_i + p_sum;
                            o_i = m_scale * o_i + p_ij.matmul(v_block);
                            m_i = m_new;

                            kv_start = kv_end;
                        }
                    }

                    if step < world_size - 1 {
                        (current_k, current_v) =
                            Self::ring_send_recv_with_comm(&current_k, &current_v, &*comm, &device);
                    }
                }

                outputs.push(o_i / l_i);
            }

            q_start = q_end;
        }

        Tensor::cat(outputs, 2)
    }

    /// Perform ring send/recv using a real communicator.
    fn ring_send_recv_with_comm<B, C>(
        k: &Tensor<B, 4>,
        v: &Tensor<B, 4>,
        comm: &C,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>)
    where
        B: Backend,
        C: Communicator<Data = TensorMessage>,
    {
        let k_shape: Vec<usize> = k.dims().to_vec();
        let v_shape: Vec<usize> = v.dims().to_vec();

        // Serialize K tensor
        let k_data = k.to_data();
        let k_values: Vec<f32> = k_data.to_vec().unwrap();
        let k_msg = TensorMessage::new(k_values, k_shape.clone());

        // Serialize V tensor
        let v_data = v.to_data();
        let v_values: Vec<f32> = v_data.to_vec().unwrap();
        let v_msg = TensorMessage::new(v_values, v_shape.clone());

        // Send K and V, receive new K and V
        // We do this in two separate send_recv calls
        let recv_k = comm.send_recv(&k_msg).expect("Failed to exchange K tensor");
        let recv_v = comm.send_recv(&v_msg).expect("Failed to exchange V tensor");

        // Deserialize received tensors
        let new_k = Tensor::<B, 4>::from_data(
            TensorData::new(recv_k.data, recv_k.shape),
            device,
        );
        let new_v = Tensor::<B, 4>::from_data(
            TensorData::new(recv_v.data, recv_v.shape),
            device,
        );

        (new_k, new_v)
    }

    fn build_causal_mask<B: Backend>(
        device: &B::Device,
        query_len: usize,
        key_len: usize,
        q_start: usize,
        kv_start: usize,
        q_offset: usize,
        kv_offset: usize,
    ) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(query_len * key_len);
        let mask_value = -1.0e4_f32;

        for i in 0..query_len {
            let absolute_pos = q_offset + q_start + i;
            for j in 0..key_len {
                let absolute_key = kv_offset + kv_start + j;
                let allowed = absolute_key <= absolute_pos;
                data.push(if allowed { 0.0 } else { mask_value });
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(data, [query_len, key_len]), device)
            .reshape([1, 1, query_len, key_len])
    }

    fn tensor_log_add_exp<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
        let max = a.clone().max_pair(b.clone());
        let diff_a = a - max.clone();
        let diff_b = b - max.clone();
        max + (diff_a.exp() + diff_b.exp()).log()
    }
}
