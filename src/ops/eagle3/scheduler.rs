/// Length scheduler for adaptive draft length selection.
///
/// Learns optimal draft length distribution from acceptance history.
#[derive(Debug, Clone)]
pub struct LengthScheduler {
    /// Acceptance rate distribution for each length: [max_length].
    length_distribution: Vec<f32>,
    /// Exponential moving average coefficient.
    ema_alpha: f32,
    /// Sample count for each length.
    sample_count: Vec<usize>,
    /// Minimum draft length.
    min_length: usize,
    /// Maximum draft length.
    max_length: usize,
}

impl LengthScheduler {
    /// Create a new length scheduler.
    #[inline(always)]
    pub fn new(min_length: usize, max_length: usize, ema_alpha: f32) -> Result<Self, &'static str> {
        if min_length == 0 {
            return Err("min_length must be > 0");
        }
        if max_length < min_length {
            return Err("max_length must be >= min_length");
        }
        if !(0.0..=1.0).contains(&ema_alpha) {
            return Err("ema_alpha must be in [0, 1]");
        }

        let length_range = max_length - min_length + 1;
        let mut length_distribution = Vec::with_capacity(length_range);
        for i in 0..length_range {
            let length = min_length + i;
            length_distribution.push(0.8f32.powi(length as i32));
        }

        Ok(Self {
            length_distribution,
            ema_alpha,
            sample_count: vec![0; length_range],
            min_length,
            max_length,
        })
    }

    /// Suggest optimal draft length based on acceptance history.
    #[inline(always)]
    pub fn suggest_length(&self) -> usize {
        let mut best_length = self.min_length;
        let mut best_expected = 0.0f32;

        for (i, &acceptance_rate) in self.length_distribution.iter().enumerate() {
            let length = self.min_length + i;
            let expected_tokens = length as f32 * acceptance_rate;

            if expected_tokens > best_expected {
                best_expected = expected_tokens;
                best_length = length;
            }
        }

        best_length
    }

    /// Update acceptance statistics for a given draft length.
    #[inline(always)]
    pub fn update(&mut self, draft_length: usize, accepted_count: usize) {
        if draft_length < self.min_length || draft_length > self.max_length {
            return;
        }

        let idx = draft_length - self.min_length;
        let acceptance_rate = accepted_count as f32 / draft_length as f32;

        self.length_distribution[idx] =
            self.ema_alpha * acceptance_rate + (1.0 - self.ema_alpha) * self.length_distribution[idx];
        self.sample_count[idx] += 1;
    }

    /// Get current acceptance rate for a specific length.
    #[inline(always)]
    pub fn get_acceptance_rate(&self, length: usize) -> Option<f32> {
        if length < self.min_length || length > self.max_length {
            return None;
        }
        Some(self.length_distribution[length - self.min_length])
    }

    /// Get total sample count across all lengths.
    #[inline(always)]
    pub fn total_samples(&self) -> usize {
        self.sample_count.iter().sum()
    }

    /// Reset acceptance statistics while preserving configuration.
    #[inline(always)]
    pub fn reset(&mut self) {
        let length_range = self.max_length - self.min_length + 1;
        self.length_distribution.clear();
        self.length_distribution.reserve(length_range);
        for i in 0..length_range {
            let length = self.min_length + i;
            self.length_distribution.push(0.8f32.powi(length as i32));
        }
        self.sample_count.clear();
        self.sample_count.resize(length_range, 0);
    }
}
