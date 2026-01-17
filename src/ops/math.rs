use crate::kernel_dispatcher::KernelFloat;

#[inline(always)]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub(crate) fn log_softmax<T: KernelFloat>(logits: &[T]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        max_val = max_val.max(v.to_f32());
    }
    if !max_val.is_finite() {
        return vec![max_val; logits.len()];
    }

    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v.to_f32() - max_val).exp();
    }
    let log_sum = max_val + sum.ln();

    logits.iter().map(|&v| v.to_f32() - log_sum).collect()
}

#[inline(always)]
pub(crate) fn find_top_token<T: KernelFloat>(logits: &[T]) -> (usize, f32) {
    let log_probs = log_softmax(logits);
    let mut best_idx = 0;
    let mut best_prob = f32::NEG_INFINITY;

    for (idx, &prob) in log_probs.iter().enumerate() {
        if prob > best_prob {
            best_prob = prob;
            best_idx = idx;
        }
    }

    (best_idx, best_prob)
}

#[inline(always)]
pub(crate) fn top_k_with_probs<T: KernelFloat>(logits: &[T], k: usize) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let log_probs = log_softmax(logits);
    let mut scored: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
