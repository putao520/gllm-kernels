//! Mamba-2 hybrid selective state space utilities.

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "mamba-kernel")]
use crate::cuda_kernels::SelectiveScanKernel as CudaSelectiveScanKernel;
#[cfg(feature = "mamba-kernel")]
use cudarc::driver::CudaContext;
#[cfg(feature = "mamba-kernel")]
use std::sync::Arc;

/// Mamba block configuration for selective state space models.
#[derive(Debug, Clone, Copy)]
pub struct MambaConfig {
    /// State space dimension.
    pub state_dim: usize,
    /// Input dimension.
    pub input_dim: usize,
    /// Expansion ratio (typical: 2).
    pub expand_ratio: usize,
    /// Enable selective gating.
    pub selective: bool,
}

/// Mamba block for selective state space modeling.
#[derive(Debug, Clone)]
pub struct MambaBlock<B: Backend> {
    /// State space dimension.
    state_dim: usize,
    /// Input dimension.
    input_dim: usize,
    /// Expansion ratio.
    expand_ratio: usize,
    _marker: PhantomData<B>,
}

/// Parameters for Mamba selective state space projection.
#[derive(Debug, Clone)]
pub struct MambaParameters<B: Backend> {
    /// Time-step projection weights: [expanded_dim, state_dim].
    pub dt_proj: Tensor<B, 2>,
    /// State decay parameters (diagonal): [state_dim].
    pub a: Tensor<B, 1>,
    /// Input projection weights: [expanded_dim, state_dim].
    pub b: Tensor<B, 2>,
    /// Output projection weights: [state_dim, expanded_dim].
    pub c: Tensor<B, 2>,
    /// Skip connection weights (diagonal): [expanded_dim].
    pub d: Tensor<B, 1>,
}

/// Stateful cache for Mamba recurrence.
#[derive(Debug, Clone)]
pub struct MambaState<B: Backend> {
    /// State tensor: [batch, state_dim].
    pub state: Tensor<B, 2>,
}

/// Hybrid strategy for mixing attention and Mamba outputs.
#[derive(Debug, Clone, Copy)]
pub enum HybridStrategy {
    /// Alternate between attention and Mamba per layer index.
    Alternating,
    /// Parallel blend with a fixed Mamba weight.
    Parallel { mamba_weight: f32 },
    /// Adaptive blend based on content energy.
    Adaptive { min_weight: f32, max_weight: f32 },
}

/// Hybrid layer for combining attention and Mamba outputs.
pub struct HybridLayer<B: Backend> {
    /// Blending strategy.
    strategy: HybridStrategy,
    /// Layer index for alternating strategy.
    layer_index: usize,
    _marker: PhantomData<B>,
}

impl MambaConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.state_dim == 0 {
            return Err("state_dim must be > 0");
        }
        if self.input_dim == 0 {
            return Err("input_dim must be > 0");
        }
        if self.expand_ratio == 0 {
            return Err("expand_ratio must be > 0");
        }
        Ok(())
    }
}

impl<B: Backend> MambaBlock<B> {
    /// Create a new Mamba block.
    pub fn new(state_dim: usize, input_dim: usize, expand_ratio: usize) -> Self {
        Self {
            state_dim,
            input_dim,
            expand_ratio,
            _marker: PhantomData,
        }
    }

    /// Create a Mamba block from configuration.
    pub fn from_config(config: &MambaConfig) -> Self {
        Self::new(config.state_dim, config.input_dim, config.expand_ratio)
    }

    /// State dimension configured for the block.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Input dimension configured for the block.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Expansion ratio configured for the block.
    pub fn expand_ratio(&self) -> usize {
        self.expand_ratio
    }

    /// Forward pass for selective state space modeling.
    ///
    /// # Shapes
    /// * `input`: [batch, seq_len, input_dim]
    /// * `state`: [batch, state_dim]
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        params: &MambaParameters<B>,
        state: Option<MambaState<B>>,
        selective: bool,
    ) -> Result<(Tensor<B, 3>, MambaState<B>), &'static str> {
        let [batch, seq_len, input_dim] = input.dims();
        if input_dim != self.input_dim {
            return Err("input dimension mismatch");
        }
        if batch == 0 || seq_len == 0 {
            return Err("input batch/seq must be > 0");
        }

        let expanded_dim = match self.input_dim.checked_mul(self.expand_ratio) {
            Some(value) => value,
            None => return Err("expanded dimension overflow"),
        };
        params.validate(expanded_dim, self.state_dim)?;

        let device = input.device();
        let input_data = input
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "input data conversion failed")?;
        let dt_proj_data = params
            .dt_proj
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "dt_proj conversion failed")?;
        let a_data = params
            .a
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "a conversion failed")?;
        let b_data = params
            .b
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "b conversion failed")?;
        let c_data = params
            .c
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "c conversion failed")?;
        let d_data = params
            .d
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|_| "d conversion failed")?;

        let mut state_data = match state {
            Some(state) => state
                .state
                .into_data()
                .into_vec::<f32>()
                .map_err(|_| "state conversion failed")?,
            None => vec![0.0f32; batch * self.state_dim],
        };
        if state_data.len() != batch * self.state_dim {
            return Err("state dimension mismatch");
        }

        let a_values: Vec<f32> = a_data.iter().map(|value| -value.exp()).collect();
        let mut output_data = vec![0.0f32; batch * seq_len * input_dim];

        for batch_idx in 0..batch {
            for time_idx in 0..seq_len {
                let input_offset = (batch_idx * seq_len + time_idx) * input_dim;
                let mut expanded_input = vec![0.0f32; expanded_dim];
                for i in 0..input_dim {
                    let value = input_data[input_offset + i];
                    for r in 0..self.expand_ratio {
                        expanded_input[i * self.expand_ratio + r] = value;
                    }
                }

                for s in 0..self.state_dim {
                    let mut dt_pre = 0.0f32;
                    let mut input_proj = 0.0f32;
                    for i in 0..expanded_dim {
                        let x = expanded_input[i];
                        dt_pre += x * dt_proj_data[i * self.state_dim + s];
                        input_proj += x * b_data[i * self.state_dim + s];
                    }
                    let mut dt = softplus(dt_pre);
                    if selective {
                        dt *= sigmoid(dt_pre);
                    }
                    let decay = (a_values[s] * dt).exp();
                    let state_idx = batch_idx * self.state_dim + s;
                    let next = state_data[state_idx] * decay + input_proj * dt;
                    state_data[state_idx] = next;
                }

                for j in 0..input_dim {
                    let mut sum = 0.0f32;
                    for r in 0..self.expand_ratio {
                        let idx = j * self.expand_ratio + r;
                        let mut y = 0.0f32;
                        for s in 0..self.state_dim {
                            y += state_data[batch_idx * self.state_dim + s]
                                * c_data[s * expanded_dim + idx];
                        }
                        y += expanded_input[idx] * d_data[idx];
                        sum += y;
                    }
                    output_data[(batch_idx * seq_len + time_idx) * input_dim + j] =
                        sum / self.expand_ratio as f32;
                }
            }
        }

        let output =
            Tensor::from_data(TensorData::new(output_data, [batch, seq_len, input_dim]), &device);
        let state =
            Tensor::from_data(TensorData::new(state_data, [batch, self.state_dim]), &device);
        Ok((output, MambaState { state }))
    }

    /// Forward pass using configuration for selective gating.
    pub fn forward_with_config(
        &self,
        input: Tensor<B, 3>,
        params: &MambaParameters<B>,
        state: Option<MambaState<B>>,
        config: &MambaConfig,
    ) -> Result<(Tensor<B, 3>, MambaState<B>), &'static str> {
        config.validate()?;
        if config.state_dim != self.state_dim
            || config.input_dim != self.input_dim
            || config.expand_ratio != self.expand_ratio
        {
            return Err("config mismatch for Mamba block");
        }
        self.forward(input, params, state, config.selective)
    }
}

impl<B: Backend> MambaParameters<B> {
    /// Validate parameter tensor shapes.
    pub fn validate(&self, expanded_dim: usize, state_dim: usize) -> Result<(), &'static str> {
        if self.dt_proj.dims() != [expanded_dim, state_dim] {
            return Err("dt_proj shape mismatch");
        }
        if self.a.dims() != [state_dim] {
            return Err("a shape mismatch");
        }
        if self.b.dims() != [expanded_dim, state_dim] {
            return Err("b shape mismatch");
        }
        if self.c.dims() != [state_dim, expanded_dim] {
            return Err("c shape mismatch");
        }
        if self.d.dims() != [expanded_dim] {
            return Err("d shape mismatch");
        }
        Ok(())
    }
}

impl<B: Backend> HybridLayer<B> {
    /// Create a new hybrid layer.
    pub fn new(strategy: HybridStrategy, layer_index: usize) -> Self {
        Self {
            strategy,
            layer_index,
            _marker: PhantomData,
        }
    }

    /// Hybrid strategy configured for this layer.
    pub fn strategy(&self) -> HybridStrategy {
        self.strategy
    }

    /// Layer index for alternating strategy.
    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    /// Combine attention and Mamba outputs.
    ///
    /// # Shapes
    /// * `attention`: [batch, seq_len, dim]
    /// * `mamba`: [batch, seq_len, dim]
    pub fn combine(
        &self,
        attention: Tensor<B, 3>,
        mamba: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, &'static str> {
        let attn_dims = attention.dims();
        let mamba_dims = mamba.dims();
        if attn_dims != mamba_dims {
            return Err("attention/mamba dimension mismatch");
        }

        match self.strategy {
            HybridStrategy::Alternating => {
                if self.layer_index % 2 == 0 {
                    Ok(attention)
                } else {
                    Ok(mamba)
                }
            }
            HybridStrategy::Parallel { mamba_weight } => {
                let weight = clamp_weight(mamba_weight);
                blend_fixed(attention, mamba, weight)
            }
            HybridStrategy::Adaptive { min_weight, max_weight } => {
                blend_adaptive(attention, mamba, min_weight, max_weight)
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

const EXP_CLAMP_MIN: f32 = -50.0;
const EXP_CLAMP_MAX: f32 = 20.0;

fn clamp_weight(weight: f32) -> f32 {
    if weight < 0.0 {
        0.0
    } else if weight > 1.0 {
        1.0
    } else {
        weight
    }
}

fn blend_fixed<B: Backend>(
    attention: Tensor<B, 3>,
    mamba: Tensor<B, 3>,
    weight: f32,
) -> Result<Tensor<B, 3>, &'static str> {
    let device = attention.device();
    let dims = attention.dims();
    let attn_data = attention
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "attention conversion failed")?;
    let mamba_data = mamba
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "mamba conversion failed")?;
    let mut output = vec![0.0f32; attn_data.len()];
    let inv = 1.0 - weight;
    for (idx, value) in output.iter_mut().enumerate() {
        *value = attn_data[idx] * inv + mamba_data[idx] * weight;
    }
    Ok(Tensor::from_data(TensorData::new(output, dims), &device))
}

fn blend_adaptive<B: Backend>(
    attention: Tensor<B, 3>,
    mamba: Tensor<B, 3>,
    min_weight: f32,
    max_weight: f32,
) -> Result<Tensor<B, 3>, &'static str> {
    let device = attention.device();
    let [batch, seq_len, dim] = attention.dims();
    let attn_data = attention
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "attention conversion failed")?;
    let mamba_data = mamba
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "mamba conversion failed")?;

    let mut output = vec![0.0f32; attn_data.len()];
    let per_batch = seq_len * dim;
    for b in 0..batch {
        let base = b * per_batch;
        let mut attn_energy = 0.0f32;
        let mut mamba_energy = 0.0f32;
        for i in 0..per_batch {
            attn_energy += attn_data[base + i].abs();
            mamba_energy += mamba_data[base + i].abs();
        }
        let denom = attn_energy + mamba_energy + 1e-6;
        let mut weight = mamba_energy / denom;
        let min_w = min_weight.min(max_weight);
        let max_w = max_weight.max(min_weight);
        if weight < min_w {
            weight = min_w;
        } else if weight > max_w {
            weight = max_w;
        }
        let inv = 1.0 - weight;
        for i in 0..per_batch {
            output[base + i] = attn_data[base + i] * inv + mamba_data[base + i] * weight;
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(output, [batch, seq_len, dim]),
        &device,
    ))
}

/// Forward selective scan using CUDA when available, falling back to CPU.
pub fn selective_scan_forward<B: Backend + 'static>(
    u: Tensor<B, 3>,
    delta: Tensor<B, 3>,
    a: Tensor<B, 2>,
    b: Tensor<B, 3>,
    c: Tensor<B, 3>,
) -> Result<Tensor<B, 3>, &'static str> {
    selective_scan_forward_with_kernel(u, delta, a, b, c, true)
}

/// Forward selective scan with explicit kernel toggle.
pub fn selective_scan_forward_with_kernel<B: Backend + 'static>(
    u: Tensor<B, 3>,
    delta: Tensor<B, 3>,
    a: Tensor<B, 2>,
    b: Tensor<B, 3>,
    c: Tensor<B, 3>,
    use_kernel: bool,
) -> Result<Tensor<B, 3>, &'static str> {
    let [batch, seq_len, expanded_dim] = u.dims();
    if delta.dims() != [batch, seq_len, expanded_dim] {
        return Err("delta shape mismatch");
    }
    let [state_dim, a_expanded] = a.dims();
    if a_expanded != expanded_dim {
        return Err("A shape mismatch");
    }
    if b.dims() != [batch, seq_len, state_dim] {
        return Err("B shape mismatch");
    }
    if c.dims() != [batch, seq_len, state_dim] {
        return Err("C shape mismatch");
    }

    if use_kernel {
        #[cfg(feature = "mamba-kernel")]
        {
            if is_cuda_backend::<B>() {
                if let Some(output) = try_forward_cuda_kernel(&u, &delta, &a, &b, &c) {
                    return Ok(output);
                }
            }
        }
    }

    let device = u.device();
    let u_data = u
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "u conversion failed")?;
    let delta_data = delta
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "delta conversion failed")?;
    let a_data = a
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "A conversion failed")?;
    let b_data = b
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "B conversion failed")?;
    let c_data = c
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| "C conversion failed")?;

    let output = selective_scan_cpu(
        &u_data,
        &delta_data,
        &a_data,
        &b_data,
        &c_data,
        batch,
        seq_len,
        state_dim,
        expanded_dim,
    )?;

    Ok(Tensor::from_data(
        TensorData::new(output, [batch, seq_len, expanded_dim]),
        &device,
    ))
}

fn selective_scan_cpu(
    u: &[f32],
    delta: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch_size: usize,
    seq_len: usize,
    state_dim: usize,
    expanded_dim: usize,
) -> Result<Vec<f32>, &'static str> {
    let expected_ud = batch_size
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(expanded_dim))
        .ok_or("u/delta length overflow")?;
    let expected_a = state_dim
        .checked_mul(expanded_dim)
        .ok_or("A length overflow")?;
    let expected_bc = batch_size
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(state_dim))
        .ok_or("B/C length overflow")?;

    if u.len() != expected_ud || delta.len() != expected_ud {
        return Err("u/delta length mismatch");
    }
    if a.len() != expected_a {
        return Err("A length mismatch");
    }
    if b.len() != expected_bc || c.len() != expected_bc {
        return Err("B/C length mismatch");
    }

    let mut output = vec![0.0f32; expected_ud];
    let per_batch_ud = seq_len * expanded_dim;
    let per_batch_bc = seq_len * state_dim;

    for b_idx in 0..batch_size {
        let base_ud = b_idx * per_batch_ud;
        let base_bc = b_idx * per_batch_bc;
        for d in 0..expanded_dim {
            let mut state = vec![0.0f32; state_dim];
            for t in 0..seq_len {
                let ud_idx = base_ud + t * expanded_dim + d;
                let dt = delta[ud_idx];
                let u_val = u[ud_idx];
                let input = dt * u_val;

                let bc_offset = base_bc + t * state_dim;
                let mut sum = 0.0f32;
                for s in 0..state_dim {
                    let a_val = a[s * expanded_dim + d];
                    let decay_arg = (dt * a_val).clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX);
                    let decay = decay_arg.exp();
                    let x = state[s] * decay + b[bc_offset + s] * input;
                    state[s] = x;
                    sum += c[bc_offset + s] * x;
                }
                output[ud_idx] = sum;
            }
        }
    }

    Ok(output)
}

#[cfg(feature = "mamba-kernel")]
fn try_forward_cuda_kernel<B: Backend + 'static>(
    u: &Tensor<B, 3>,
    delta: &Tensor<B, 3>,
    a: &Tensor<B, 2>,
    b: &Tensor<B, 3>,
    c: &Tensor<B, 3>,
) -> Option<Tensor<B, 3>>
where
    B::FloatElem: 'static,
{
    use std::any::{Any, TypeId};

    let [batch, seq_len, expanded_dim] = u.dims();
    let [state_dim, a_expanded] = a.dims();
    if a_expanded != expanded_dim {
        log::warn!("CUDA selective scan fallback: A shape mismatch");
        return None;
    }
    if delta.dims() != [batch, seq_len, expanded_dim] {
        log::warn!("CUDA selective scan fallback: delta shape mismatch");
        return None;
    }
    if b.dims() != [batch, seq_len, state_dim] {
        log::warn!("CUDA selective scan fallback: B shape mismatch");
        return None;
    }
    if c.dims() != [batch, seq_len, state_dim] {
        log::warn!("CUDA selective scan fallback: C shape mismatch");
        return None;
    }

    let device = u.device();
    let cuda_index = {
        #[cfg(feature = "cuda")]
        {
            let device_any = &device as &dyn Any;
            if let Some(cuda_device) = device_any.downcast_ref::<burn_cuda::CudaDevice>() {
                cuda_device.index
            } else {
                0
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    };

    let cuda_ctx = match CudaContext::new(cuda_index) {
        Ok(ctx) => Arc::new(ctx),
        Err(err) => {
            log::warn!("CUDA selective scan fallback: device init failed: {err}");
            return None;
        }
    };
    let stream = cuda_ctx.default_stream();
    let kernel = match CudaSelectiveScanKernel::new(&cuda_ctx) {
        Ok(kernel) => kernel,
        Err(err) => {
            log::warn!("CUDA selective scan fallback: kernel load failed: {err}");
            return None;
        }
    };

    let elem_type = TypeId::of::<B::FloatElem>();
    if elem_type != TypeId::of::<f32>() {
        log::warn!("CUDA selective scan fallback: unsupported dtype");
        return None;
    }

    let u_host = u.clone().into_data().into_vec::<f32>().ok()?;
    let delta_host = delta.clone().into_data().into_vec::<f32>().ok()?;
    let a_host = a.clone().into_data().into_vec::<f32>().ok()?;
    let b_host = b.clone().into_data().into_vec::<f32>().ok()?;
    let c_host = c.clone().into_data().into_vec::<f32>().ok()?;

    let u_dev = stream.clone_htod(&u_host).ok()?;
    let delta_dev = stream.clone_htod(&delta_host).ok()?;
    let a_dev = stream.clone_htod(&a_host).ok()?;
    let b_dev = stream.clone_htod(&b_host).ok()?;
    let c_dev = stream.clone_htod(&c_host).ok()?;

    let output = kernel
        .forward(
            &stream,
            &u_dev,
            &delta_dev,
            &a_dev,
            &b_dev,
            &c_dev,
            batch,
            seq_len,
            state_dim,
            expanded_dim,
        )
        .ok()?;

    let out_host = stream.clone_dtoh(&output).ok()?;
    Some(Tensor::<B, 3>::from_data(
        TensorData::new(out_host, [batch, seq_len, expanded_dim]),
        &device,
    ))
}

#[cfg(feature = "mamba-kernel")]
fn is_cuda_backend<B: Backend + 'static>() -> bool {
    #[cfg(feature = "cuda")]
    {
        use std::any::TypeId;

        let type_id = TypeId::of::<B>();
        if type_id == TypeId::of::<burn_cuda::Cuda>() {
            return true;
        }
        #[cfg(feature = "fusion")]
        {
            if type_id == TypeId::of::<burn_fusion::Fusion<burn_cuda::Cuda>>() {
                return true;
            }
        }
        false
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_mamba_forward_shapes() {
        let config = MambaConfig {
            state_dim: 2,
            input_dim: 3,
            expand_ratio: 2,
            selective: true,
        };
        let block = MambaBlock::<NdArray<f32>>::from_config(&config);
        let device = <NdArray<f32> as Backend>::Device::default();
        let input = Tensor::from_data(
            TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 2, 3]),
            &device,
        );
        let params = MambaParameters {
            dt_proj: Tensor::from_data(TensorData::new(vec![0.05; 12], [6, 2]), &device),
            a: Tensor::from_data(TensorData::new(vec![0.1, 0.2], [2]), &device),
            b: Tensor::from_data(TensorData::new(vec![0.02; 12], [6, 2]), &device),
            c: Tensor::from_data(TensorData::new(vec![0.03; 12], [2, 6]), &device),
            d: Tensor::from_data(TensorData::new(vec![0.1; 6], [6]), &device),
        };

        let (output, state) = block
            .forward_with_config(input, &params, None, &config)
            .expect("forward");
        assert_eq!(output.dims(), [1, 2, 3]);
        assert_eq!(state.state.dims(), [1, 2]);
    }

    #[test]
    fn test_hybrid_parallel_blend() {
        let layer = HybridLayer::<NdArray<f32>>::new(
            HybridStrategy::Parallel { mamba_weight: 0.25 },
            0,
        );
        let device = <NdArray<f32> as Backend>::Device::default();
        let attention =
            Tensor::from_data(TensorData::new(vec![1.0, 3.0], [1, 1, 2]), &device);
        let mamba = Tensor::from_data(TensorData::new(vec![5.0, 1.0], [1, 1, 2]), &device);

        let output = layer.combine(attention, mamba).expect("combine");
        let data = output
            .into_data()
            .into_vec::<f32>()
            .expect("output data");
        assert!((data[0] - 2.0).abs() < 1e-4);
        assert!((data[1] - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_selective_scan_shapes() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let u = Tensor::from_data(
            TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 2, 3]),
            &device,
        );
        let delta = Tensor::from_data(
            TensorData::new(vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [1, 2, 3]),
            &device,
        );
        let a = Tensor::from_data(TensorData::new(vec![0.1; 6], [2, 3]), &device);
        let b = Tensor::from_data(TensorData::new(vec![0.2; 4], [1, 2, 2]), &device);
        let c = Tensor::from_data(TensorData::new(vec![0.3; 4], [1, 2, 2]), &device);

        let output = selective_scan_forward(u, delta, a, b, c).expect("selective scan");
        assert_eq!(output.dims(), [1, 2, 3]);
    }
}
