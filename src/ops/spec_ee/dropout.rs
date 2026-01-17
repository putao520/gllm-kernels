/// Layer dropout rate function.
///
/// Following LayerSkip paper: low layers have low dropout, high layers have high dropout.
#[derive(Debug, Clone, Copy)]
pub struct LayerDropoutSchedule {
    /// Minimum dropout rate (for layer 0).
    pub min_rate: f32,
    /// Maximum dropout rate (for last layer).
    pub max_rate: f32,
    /// Total number of layers.
    pub num_layers: usize,
}

impl LayerDropoutSchedule {
    /// Create a linear dropout schedule.
    #[inline(always)]
    pub fn linear(min_rate: f32, max_rate: f32, num_layers: usize) -> Result<Self, &'static str> {
        if num_layers == 0 {
            return Err("num_layers must be > 0");
        }
        Ok(Self {
            min_rate: min_rate.clamp(0.0, 1.0),
            max_rate: max_rate.clamp(0.0, 1.0),
            num_layers,
        })
    }

    /// Get dropout rate for a specific layer.
    #[inline(always)]
    pub fn get_rate(&self, layer_idx: usize) -> f32 {
        if self.num_layers <= 1 {
            return self.min_rate;
        }
        let t = layer_idx as f32 / (self.num_layers - 1) as f32;
        self.min_rate + t * (self.max_rate - self.min_rate)
    }

    /// Check if a layer should be dropped (training only).
    #[inline(always)]
    pub fn should_drop(&self, layer_idx: usize, random_value: f32) -> bool {
        random_value < self.get_rate(layer_idx)
    }
}
