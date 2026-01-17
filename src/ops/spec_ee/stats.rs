/// Statistics for SpecEE inference.
#[derive(Debug, Clone, Default)]
pub struct SpecEEStats {
    /// Exit counts per layer.
    pub exit_layer_counts: Vec<usize>,
    /// Total early exits.
    pub total_early_exits: usize,
    /// Total full forwards.
    pub total_full_forwards: usize,
    /// Acceptance count (early exit matched full forward).
    pub accepted_count: usize,
    /// Rejection count (early exit differed from full forward).
    pub rejected_count: usize,
    /// Average exit layer.
    pub avg_exit_layer: f32,
    /// Acceptance rate.
    pub acceptance_rate: f32,
}

impl SpecEEStats {
    /// Initialize with number of layers.
    #[inline(always)]
    pub fn new(num_layers: usize) -> Self {
        Self {
            exit_layer_counts: vec![0; num_layers],
            ..Default::default()
        }
    }

    /// Update stats after an early exit.
    #[inline(always)]
    pub fn record_early_exit(&mut self, exit_layer: usize, accepted: bool) {
        if exit_layer < self.exit_layer_counts.len() {
            self.exit_layer_counts[exit_layer] += 1;
        }
        self.total_early_exits += 1;

        if accepted {
            self.accepted_count += 1;
        } else {
            self.rejected_count += 1;
        }

        self.update_derived();
    }

    /// Update stats after a full forward.
    #[inline(always)]
    pub fn record_full_forward(&mut self) {
        self.total_full_forwards += 1;
        self.update_derived();
    }

    #[inline(always)]
    fn update_derived(&mut self) {
        let total_exits: usize = self.exit_layer_counts.iter().sum();
        if total_exits > 0 {
            let weighted_sum: usize = self.exit_layer_counts
                .iter()
                .enumerate()
                .map(|(i, &count)| i * count)
                .sum();
            self.avg_exit_layer = weighted_sum as f32 / total_exits as f32;
        }

        let total_attempts = self.accepted_count + self.rejected_count;
        if total_attempts > 0 {
            self.acceptance_rate = self.accepted_count as f32 / total_attempts as f32;
        }
    }

    /// Estimate speedup based on average exit layer.
    #[inline(always)]
    pub fn estimated_speedup(&self, num_layers: usize) -> f32 {
        if self.total_early_exits == 0 {
            return 1.0;
        }
        let exit_ratio = num_layers as f32 / (self.avg_exit_layer + 1.0);
        let effective_rate = self.acceptance_rate * 0.9 + 0.1;
        exit_ratio * effective_rate
    }
}
