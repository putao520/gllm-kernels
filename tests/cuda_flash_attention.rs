#[cfg(all(feature = "cuda-kernel", feature = "cpu"))]
mod cuda_flash_attention_tests {
    use std::sync::Arc;

    use burn::tensor::backend::Backend;
    use burn::tensor::{Distribution, Tensor};
    use burn_ndarray::NdArray;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
    use gllm_kernels::{FlashAttentionConfig, FlashAttentionKernel, HierarchicalFlashAttention};
    use half::f16;

    type CpuBackend = NdArray<f32>;

    fn cpu_attention_output(
        q: Tensor<CpuBackend, 4>,
        k: Tensor<CpuBackend, 4>,
        v: Tensor<CpuBackend, 4>,
        causal: bool,
    ) -> Vec<f32> {
        let attention = HierarchicalFlashAttention::new(FlashAttentionConfig::default());
        let output = attention.forward(q, k, v, causal, 0);
        output
            .into_data()
            .into_vec::<f32>()
            .expect("cpu output")
    }

    #[test]
    fn test_cuda_flash_attention_f32_matches_cpu() {
        let device = <CpuBackend as Backend>::Device::default();
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 8;
        let causal = true;

        let q = Tensor::<CpuBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<CpuBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<CpuBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );

        let cpu_output = cpu_attention_output(q.clone(), k.clone(), v.clone(), causal);
        let q_host = q
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("q host data");
        let k_host = k
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("k host data");
        let v_host = v
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("v host data");

        let cuda_ctx = match CudaContext::new(0) {
            Ok(ctx) => Arc::new(ctx),
            Err(_) => return,
        };
        let stream: Arc<CudaStream> = cuda_ctx.default_stream();
        let kernel = match FlashAttentionKernel::new(&cuda_ctx) {
            Ok(kernel) => kernel,
            Err(_) => return,
        };

        let q_dev: CudaSlice<f32> = stream.clone_htod(&q_host).expect("q copy");
        let k_dev: CudaSlice<f32> = stream.clone_htod(&k_host).expect("k copy");
        let v_dev: CudaSlice<f32> = stream.clone_htod(&v_host).expect("v copy");

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = kernel
            .forward_f32(
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                causal,
                scale,
                0,
            )
            .expect("cuda kernel output");

        let gpu_output = stream.clone_dtoh(&output).expect("gpu output copy");

        for (idx, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
            let diff = (cpu - gpu).abs();
            assert!(diff < 1e-2, "f32 mismatch at {idx}: cpu={cpu}, gpu={gpu}");
        }
    }

    #[test]
    fn test_cuda_flash_attention_f16_matches_cpu() {
        let device = <CpuBackend as Backend>::Device::default();
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 8;
        let causal = true;

        let q = Tensor::<CpuBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let k = Tensor::<CpuBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let v = Tensor::<CpuBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            Distribution::Normal(0.0, 0.5),
            &device,
        );

        let cpu_output = cpu_attention_output(q.clone(), k.clone(), v.clone(), causal);
        let q_host_f32 = q
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("q host data");
        let k_host_f32 = k
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("k host data");
        let v_host_f32 = v
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("v host data");

        let q_host: Vec<f16> = q_host_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let k_host: Vec<f16> = k_host_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let v_host: Vec<f16> = v_host_f32.iter().map(|v| f16::from_f32(*v)).collect();

        let cuda_ctx = match CudaContext::new(0) {
            Ok(ctx) => Arc::new(ctx),
            Err(_) => return,
        };
        let stream: Arc<CudaStream> = cuda_ctx.default_stream();
        let kernel = match FlashAttentionKernel::new(&cuda_ctx) {
            Ok(kernel) => kernel,
            Err(_) => return,
        };

        let q_dev: CudaSlice<f16> = stream.clone_htod(&q_host).expect("q copy");
        let k_dev: CudaSlice<f16> = stream.clone_htod(&k_host).expect("k copy");
        let v_dev: CudaSlice<f16> = stream.clone_htod(&v_host).expect("v copy");

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = kernel
            .forward_f16(
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                causal,
                scale,
                0,
            )
            .expect("cuda kernel output");

        let gpu_output = stream.clone_dtoh(&output).expect("gpu output copy");

        for (idx, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
            let gpu_val = f32::from(*gpu);
            let diff = (cpu - gpu_val).abs();
            assert!(diff < 5e-2, "f16 mismatch at {idx}: cpu={cpu}, gpu={gpu_val}");
        }
    }
}
