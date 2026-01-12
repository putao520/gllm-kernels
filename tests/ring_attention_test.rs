#[cfg(feature = "cpu")]
use burn::tensor::{backend::Backend, Distribution, Tensor};
#[cfg(feature = "cpu")]
use burn_ndarray::NdArray;
#[cfg(feature = "cpu")]
use gllm_kernels::{
    AttentionConfig, CommBackend, HierarchicalFlashAttention, RingAttention, RingAttentionConfig,
};

#[cfg(feature = "cpu")]
type TestBackend = NdArray<f32>;

#[cfg(feature = "cpu")]
fn build_tensors(
    device: &<TestBackend as Backend>::Device,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> (
    Tensor<TestBackend, 4>,
    Tensor<TestBackend, 4>,
    Tensor<TestBackend, 4>,
) {
    let q = Tensor::random(
        [batch, heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let k = Tensor::random(
        [batch, heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let v = Tensor::random(
        [batch, heads, seq_len, head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    (q, k, v)
}

#[cfg(feature = "cpu")]
#[test]
fn test_ring_attention_single_node() {
    let device = <TestBackend as Backend>::Device::default();
    let batch = 1;
    let heads = 2;
    let seq_len = 8;
    let head_dim = 4;

    let (q, k, v) = build_tensors(&device, batch, heads, seq_len, head_dim);

    let ring_config = RingAttentionConfig {
        world_size: 1,
        rank: 0,
        local_seq_len: seq_len,
        causal: false,
        comm_backend: CommBackend::SharedMemory,
    };
    let ring = RingAttention::new(ring_config);

    let config = AttentionConfig::new(batch, heads, seq_len, seq_len, head_dim).with_causal(false);
    let output_ring = ring.forward(q.clone(), k.clone(), v.clone(), &config);

    let baseline = HierarchicalFlashAttention::default_config();
    let output_ref = baseline.forward(q, k, v, false, 0);

    let ring_data = output_ring
        .into_data()
        .into_vec::<f32>()
        .expect("ring output");
    let ref_data = output_ref
        .into_data()
        .into_vec::<f32>()
        .expect("reference output");

    for (i, (r, b)) in ring_data.iter().zip(ref_data.iter()).enumerate() {
        let diff = (r - b).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at {}: ring={}, ref={}, diff={}",
            i,
            r,
            b,
            diff
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_ring_attention_causal_mask() {
    let device = <TestBackend as Backend>::Device::default();
    let batch = 1;
    let heads = 2;
    let seq_len = 8;
    let head_dim = 4;

    let (q, k, v) = build_tensors(&device, batch, heads, seq_len, head_dim);

    let ring_config = RingAttentionConfig {
        world_size: 1,
        rank: 0,
        local_seq_len: seq_len,
        causal: true,
        comm_backend: CommBackend::SharedMemory,
    };
    let ring = RingAttention::new(ring_config);

    let config = AttentionConfig::new(batch, heads, seq_len, seq_len, head_dim).with_causal(true);
    let output_ring = ring.forward(q.clone(), k.clone(), v.clone(), &config);

    let baseline = HierarchicalFlashAttention::default_config();
    let output_ref = baseline.forward(q, k, v, true, 0);

    let ring_data = output_ring
        .into_data()
        .into_vec::<f32>()
        .expect("ring output");
    let ref_data = output_ref
        .into_data()
        .into_vec::<f32>()
        .expect("reference output");

    for (i, (r, b)) in ring_data.iter().zip(ref_data.iter()).enumerate() {
        let diff = (r - b).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at {}: ring={}, ref={}, diff={}",
            i,
            r,
            b,
            diff
        );
    }
}
