//! Mamba-2 hybrid selective state space utilities.

use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

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
}
