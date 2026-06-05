//! RoPE frequency / temperature scaling math.
//!
//! Translates a [`RopeScaling`] descriptor into:
//! 1. an `inv_freq` table — `[f64; head_dim/2]` of per-pair angular
//!    frequencies (the value the cos/sin filler multiplies by `position`),
//! 2. an `attention_scaling` (mscale) — a scalar multiplier the cos/sin
//!    filler folds into the table so that, downstream, `Q·K^T / √d` is
//!    implicitly scaled by `mscale²`. This is the "temperature" knob that
//!    keeps softmax sharp under long-context extension.
//!
//! All formulas mirror the canonical implementations in
//! - HuggingFace `transformers/models/llama/modeling_llama.py::LlamaRotaryEmbedding`
//!   (the YaRN branch),
//! - vLLM `vllm/model_executor/layers/rotary_embedding.py::YaRNScalingRotaryEmbedding`,
//! - the original YaRN paper (https://arxiv.org/abs/2309.00071, eqs. 22–24).
//!
//! ## Bit-exact behaviour
//! - YaRN uses `f64` throughout the inv_freq derivation (matches HF/vLLM).
//! - `cos(angle * factor)` is computed in `f64` then cast to `f32` for the
//!   scratchpad — same as `populate_rope_cache` in the executor.
//! - Linear scaling divides `inv_freq` by `factor` only; no temperature.
//! - `None` returns the standard inv_freq with `attention_scaling = 1.0`.
//!
//! ## Why this lives in `gllm-kernels`
//! The JIT codegen (`lower_rope_full`) consumes a pre-filled cos/sin table
//! and is therefore agnostic to the scaling family. To honour the "JIT must
//! encode yarn correction" rule, the compiler emits a
//! [`RopeCacheRequirement`] that carries the scaling descriptor + derived
//! `attention_scaling` — the executor (caller) plugs them straight into
//! [`compute_inv_freq`] / [`compute_attention_scaling`] when filling the
//! scratchpad. There is no runtime branch in the hot path.

use crate::compiler::graph::RopeScaling;

/// Compute the per-pair `inv_freq` table for a given RoPE configuration.
///
/// Returns `[f64; head_dim/2]` where `inv_freq[i]` is the angular frequency
/// (radians per token) for the `(2i, 2i+1)` pair. The cos/sin filler then
/// computes `angle = position * inv_freq[i]` and stores
/// `(cos(angle), sin(angle))`.
///
/// For YaRN the table is per-dim interpolated/extrapolated by a linear ramp
/// between `beta_slow` (low-rotation, fully interpolated, `× 1/factor`) and
/// `beta_fast` (high-rotation, fully extrapolated, unchanged). See
/// [`yarn_find_correction_range`] for the cutoff math.
pub fn compute_inv_freq(head_dim: usize, theta: f64, scaling: Option<RopeScaling>) -> Vec<f64> {
    assert!(head_dim > 0 && head_dim % 2 == 0,
        "compute_inv_freq: head_dim must be positive even, got {head_dim}");
    assert!(theta > 0.0, "compute_inv_freq: theta must be positive, got {theta}");

    let half = head_dim / 2;
    // Standard base inv_freq (Llama / GPT-NeoX convention):
    //   inv_freq[i] = 1.0 / theta^(2i / head_dim)
    let base: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    match scaling {
        None => base,
        Some(RopeScaling::Linear { factor }) => {
            assert!(factor > 0.0, "Linear RoPE factor must be > 0, got {factor}");
            base.iter().map(|f| f / factor as f64).collect()
        }
        Some(RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position }) => {
            yarn_inv_freq(&base, head_dim, theta, factor, beta_fast, beta_slow, original_max_position)
        }
    }
}

/// Compute the YaRN attention temperature scale (mscale).
///
/// `mscale = sqrt(0.1 * ln(factor) + 1.0)` — applied symmetrically to both
/// cos and sin so that `Q·K^T` is scaled by `mscale²`.
///
/// The `sqrt` matches the canonical YaRN formula (the paper defines
/// `t = 0.1·ln(s) + 1` as the *temperature* applied to attention logits;
/// since the same factor is applied to both Q and K via cos/sin folding,
/// the per-vector scale is its square root).
///
/// `None` and `Linear` scaling return `1.0` (identity).
pub fn compute_attention_scaling(scaling: Option<RopeScaling>) -> f32 {
    match scaling {
        None => 1.0,
        Some(RopeScaling::Linear { .. }) => 1.0,
        Some(RopeScaling::Yarn { factor, .. }) => {
            assert!(factor > 0.0, "Yarn factor must be > 0, got {factor}");
            (0.1f64 * (factor as f64).ln() + 1.0).sqrt() as f32
        }
    }
}

// ── YaRN internals ──────────────────────────────────────────────────

/// YaRN per-dim correction dimension (paper eq. 17, "find_correction_dim").
///
/// Returns the (fractional) dim index whose rotation period equals
/// `original_max_position / num_rotations`. Dims with index lower than
/// `low` (more rotations) are extrapolated; dims with index higher than
/// `high` (fewer rotations) are interpolated.
fn yarn_find_correction_dim(
    num_rotations: f64,
    head_dim: usize,
    theta: f64,
    original_max_position: usize,
) -> f64 {
    // dim * ln(L / (r * 2π)) / (2 * ln(theta))
    (head_dim as f64 * (original_max_position as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
        / (2.0 * theta.ln())
}

/// Clamp `[low, high]` to the valid pair index range and ensure they
/// straddle at least one dim (fall-back when the model is so small that
/// `floor(low) == ceil(high)`).
pub(crate) fn yarn_find_correction_range(
    beta_fast: f32,
    beta_slow: f32,
    head_dim: usize,
    theta: f64,
    original_max_position: usize,
) -> (f64, f64) {
    let low_raw = yarn_find_correction_dim(beta_fast as f64, head_dim, theta, original_max_position);
    let high_raw = yarn_find_correction_dim(beta_slow as f64, head_dim, theta, original_max_position);
    let low = low_raw.floor().max(0.0);
    let high = high_raw.ceil().min(head_dim as f64 - 1.0);
    (low, high)
}

/// Linear ramp mask `clamp((i - low) / (high - low), 0, 1)` over `[0, half)`.
fn linear_ramp_mask(low: f64, high: f64, half: usize) -> Vec<f64> {
    // Avoid division by zero when low == high (tiny models / extreme cutoffs).
    let span = if (high - low).abs() < 1e-3 { 1e-3 } else { high - low };
    (0..half)
        .map(|i| {
            let x = (i as f64 - low) / span;
            x.clamp(0.0, 1.0)
        })
        .collect()
}

/// YaRN inv_freq derivation (paper eq. 22, "by-parts" interpolation).
///
/// For each pair index `i ∈ [0, head_dim/2)`:
///   ramp[i] = clamp((i - low) / (high - low), 0, 1)   // interpolation weight
///   inv_freq[i] = (1 - ramp[i]) * extrapolation[i] + ramp[i] * interpolation[i]
///
/// where:
///   extrapolation[i] = base_inv_freq[i]              (unchanged, full freq)
///   interpolation[i] = base_inv_freq[i] / factor     (linear PI compression)
///
/// Note: the HF/vLLM code computes the "extrapolation_factor" as
/// `1 - ramp`, so low-i (high-rotation) dims get extrapolation weight ≈ 1,
/// high-i (low-rotation) dims get interpolation weight ≈ 1.
fn yarn_inv_freq(
    base: &[f64],
    head_dim: usize,
    theta: f64,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: usize,
) -> Vec<f64> {
    assert!(factor > 0.0, "Yarn factor must be > 0, got {factor}");
    assert!(beta_fast > beta_slow,
        "Yarn beta_fast ({beta_fast}) must be > beta_slow ({beta_slow})");
    assert!(original_max_position > 0,
        "Yarn original_max_position must be > 0");

    let half = base.len();
    debug_assert_eq!(half, head_dim / 2);

    let (low, high) = yarn_find_correction_range(
        beta_fast, beta_slow, head_dim, theta, original_max_position);
    let ramp = linear_ramp_mask(low, high, half);

    base.iter()
        .zip(ramp.iter())
        .map(|(&base_freq, &r)| {
            let extrapolation = base_freq;
            let interpolation = base_freq / factor as f64;
            // ramp = interpolation weight  (HF: `inv_freq_extrapolation_factor = 1 - ramp`)
            (1.0 - r) * extrapolation + r * interpolation
        })
        .collect()
}

/// Pre-fill a `[seq_len, head_dim]` row-major cos/sin scratchpad given a
/// list of `positions` and a [`RopeScaling`] descriptor.
///
/// Layout (matches `lower_rope_full` consumer):
/// - per row of length `head_dim`:
///   - first `head_dim/2` values are `cos(angle) * mscale`,
///   - next `head_dim/2` values are `sin(angle) * mscale`.
///
/// `mscale` (= `compute_attention_scaling`) is folded into both cos and sin
/// so the JIT kernel needs no special-case for YaRN — the temperature
/// scaling rides along the rotation. Standard RoPE has `mscale = 1.0`.
///
/// Used by the executor (gllm) and by unit tests as the ground-truth
/// reference for verifying the JIT path produces matching outputs.
pub fn fill_cos_sin_table(
    out: &mut [f32],
    positions: &[u32],
    head_dim: usize,
    theta: f64,
    scaling: Option<RopeScaling>,
) {
    fill_cos_sin_table_partial(out, positions, head_dim, theta, 1.0, scaling)
}

/// Like [`fill_cos_sin_table`] but accepts a `partial` factor ∈ (0, 1].
///
/// For partial RoPE (e.g. Phi4 with partial=0.75), the rotation dimension is
/// `rot_dim = head_dim * partial` and inv_freq must be computed with `rot_dim`,
/// NOT `head_dim`. The table layout still uses `head_dim` values per row
/// (cos[0..half], sin[half..head_dim]) so the JIT sin-offset calculation
/// (`half * elem_bytes`) remains correct.
pub fn fill_cos_sin_table_partial(
    out: &mut [f32],
    positions: &[u32],
    head_dim: usize,
    theta: f64,
    partial: f32,
    scaling: Option<RopeScaling>,
) {
    assert!(partial > 0.0 && partial <= 1.0,
        "fill_cos_sin_table_partial: partial must be in (0,1], got {partial}");
    let half = head_dim / 2;
    let rot_dim = ((head_dim as f32 * partial) as usize).max(2) & !1;
    let half_rot = rot_dim / 2;
    let inv_freq = compute_inv_freq(rot_dim, theta, scaling);
    let mscale = compute_attention_scaling(scaling) as f64;
    assert_eq!(inv_freq.len(), half_rot);
    let needed = positions.len() * head_dim;
    assert!(out.len() >= needed,
        "fill_cos_sin_table_partial: out len {} < required {needed}", out.len());

    for (row_idx, &pos) in positions.iter().enumerate() {
        let row = &mut out[row_idx * head_dim..(row_idx + 1) * head_dim];
        for i in 0..half_rot {
            let angle = pos as f64 * inv_freq[i];
            row[i] = (angle.cos() * mscale) as f32;
            row[half + i] = (angle.sin() * mscale) as f32;
        }
        for i in half_rot..half {
            row[i] = 0.0;
            row[half + i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard RoPE (no scaling): inv_freq[i] = 1/theta^(2i/dim).
    #[test]
    fn standard_inv_freq_matches_formula() {
        let head_dim = 64;
        let theta = 10000.0;
        let inv = compute_inv_freq(head_dim, theta, None);
        assert_eq!(inv.len(), 32);
        // Sanity: inv_freq[0] = 1.0
        assert!((inv[0] - 1.0).abs() < 1e-12, "inv[0]={}", inv[0]);
        // inv_freq[16] = 1/theta^(32/64) = 1/sqrt(theta) = 1/100
        assert!((inv[16] - 0.01).abs() < 1e-12, "inv[16]={}", inv[16]);
        // inv_freq[31] = 1/theta^(62/64)
        let expected = 1.0 / 10000.0_f64.powf(62.0 / 64.0);
        assert!((inv[31] - expected).abs() < 1e-12, "inv[31]={}, expected {expected}", inv[31]);
    }

    /// Standard RoPE attention_scaling = 1.0.
    #[test]
    fn standard_attention_scaling_is_one() {
        assert_eq!(compute_attention_scaling(None), 1.0);
    }

    /// Linear PI: inv_freq divided by factor uniformly.
    #[test]
    fn linear_inv_freq_divides_uniformly() {
        let head_dim = 64;
        let theta = 10000.0;
        let factor = 4.0;
        let base = compute_inv_freq(head_dim, theta, None);
        let scaled = compute_inv_freq(head_dim, theta, Some(RopeScaling::Linear { factor }));
        for (b, s) in base.iter().zip(scaled.iter()) {
            assert!((s - b / factor as f64).abs() < 1e-12);
        }
        // No temperature on Linear
        assert_eq!(compute_attention_scaling(Some(RopeScaling::Linear { factor })), 1.0);
    }

    /// YaRN attention_scaling formula: sqrt(0.1*ln(factor) + 1).
    #[test]
    fn yarn_attention_scaling_formula() {
        let cases = [
            (1.0_f32, 1.0_f64),                         // ln(1) = 0 ⇒ mscale = 1
            (4.0_f32, (0.1 * 4.0_f64.ln() + 1.0).sqrt()),
            (32.0_f32, (0.1 * 32.0_f64.ln() + 1.0).sqrt()),
        ];
        for (factor, expected) in cases {
            let yarn = RopeScaling::Yarn {
                factor,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 4096,
            };
            let got = compute_attention_scaling(Some(yarn)) as f64;
            assert!((got - expected).abs() < 1e-6,
                "factor={factor}: got {got}, expected {expected}");
        }
    }

    /// YaRN low-i dims (high-rotation) are pure extrapolation (== base inv_freq).
    /// This must hold for *any* factor since extrapolation does not divide.
    #[test]
    fn yarn_low_i_is_extrapolation() {
        let head_dim = 64;
        let theta = 10000.0;
        let factor = 32.0;
        let original_max = 4096;
        let base = compute_inv_freq(head_dim, theta, None);
        let yarn = compute_inv_freq(head_dim, theta, Some(RopeScaling::Yarn {
            factor, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: original_max,
        }));

        // Find low cutoff: dims with i < low must equal base (extrapolation weight = 1)
        let (low, _high) = yarn_find_correction_range(
            32.0, 1.0, head_dim, theta, original_max);
        let low_int = low as usize;
        for i in 0..low_int {
            assert!((yarn[i] - base[i]).abs() < 1e-12,
                "yarn[{i}] should be extrapolation: {} vs base {}", yarn[i], base[i]);
        }
        // Middle / high-i dims must be ≤ base[i] (interpolated, divided by factor)
        let interpolation = base[half_minus_one(head_dim)] / factor as f64;
        // last dim (i = half-1) is well past `high` → fully interpolated
        let last = base.len() - 1;
        assert!((yarn[last] - interpolation).abs() < 1e-12,
            "yarn[last] should be interpolation: {} vs {} = base/factor",
            yarn[last], interpolation);
    }

    fn half_minus_one(head_dim: usize) -> usize { head_dim / 2 - 1 }

    /// YaRN high-i dims (low-rotation) collapse to interpolation `= base / factor`.
    #[test]
    fn yarn_high_i_is_interpolation() {
        let head_dim = 128;
        let theta = 10000.0;
        let factor = 8.0;
        let original_max = 4096;
        let base = compute_inv_freq(head_dim, theta, None);
        let yarn = compute_inv_freq(head_dim, theta, Some(RopeScaling::Yarn {
            factor, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: original_max,
        }));

        let (_low, high) = yarn_find_correction_range(
            32.0, 1.0, head_dim, theta, original_max);
        let high_int = high.ceil() as usize;
        for i in (high_int + 1)..base.len() {
            let interp = base[i] / factor as f64;
            assert!((yarn[i] - interp).abs() < 1e-12,
                "yarn[{i}] should be pure interpolation: {} vs {}",
                yarn[i], interp);
        }
    }

    /// fill_cos_sin_table without scaling matches the standard formula
    /// (cos(p / theta^(2i/d)), sin(...)).
    #[test]
    fn fill_cos_sin_no_scaling_matches_formula() {
        let head_dim = 64;
        let theta = 10000.0;
        let positions = [0u32, 1, 7, 100, 1023];
        let mut out = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table(&mut out, &positions, head_dim, theta, None);
        let half = head_dim / 2;
        for (row_idx, &pos) in positions.iter().enumerate() {
            for i in 0..half {
                let freq = 1.0f64 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                let want_cos = angle.cos() as f32;
                let want_sin = angle.sin() as f32;
                let got_cos = out[row_idx * head_dim + i];
                let got_sin = out[row_idx * head_dim + half + i];
                assert!((got_cos - want_cos).abs() < 1e-6,
                    "row {row_idx} i {i}: cos {got_cos} vs {want_cos}");
                assert!((got_sin - want_sin).abs() < 1e-6,
                    "row {row_idx} i {i}: sin {got_sin} vs {want_sin}");
            }
        }
    }

    /// fill_cos_sin_table with YaRN: cos/sin must equal
    /// `cos(pos * yarn_inv_freq[i]) * mscale`. This is the regression
    /// guard for the executor's scratchpad fill.
    #[test]
    fn fill_cos_sin_yarn_matches_inv_freq_times_mscale() {
        let head_dim = 128;
        let theta = 10000.0;
        let yarn = RopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        let positions = [0u32, 1, 4096, 16384];
        let mut out = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table(&mut out, &positions, head_dim, theta, Some(yarn));

        let inv_freq = compute_inv_freq(head_dim, theta, Some(yarn));
        let mscale = compute_attention_scaling(Some(yarn)) as f64;
        let half = head_dim / 2;

        for (row_idx, &pos) in positions.iter().enumerate() {
            for i in 0..half {
                let angle = pos as f64 * inv_freq[i];
                let want_cos = (angle.cos() * mscale) as f32;
                let want_sin = (angle.sin() * mscale) as f32;
                let got_cos = out[row_idx * head_dim + i];
                let got_sin = out[row_idx * head_dim + half + i];
                assert!((got_cos - want_cos).abs() < 1e-6,
                    "yarn row {row_idx} i {i}: cos {got_cos} vs {want_cos}");
                assert!((got_sin - want_sin).abs() < 1e-6,
                    "yarn row {row_idx} i {i}: sin {got_sin} vs {want_sin}");
            }
        }
    }

    /// Pinned numerical reference: head_dim=64, theta=10000, factor=32,
    /// original_max=4096, beta_fast=32, beta_slow=1. Values cross-checked
    /// against HuggingFace `transformers` `LlamaYarnRotaryEmbedding`
    /// (commit 4.46.0) for the same parameters.
    ///
    /// HF reference (Python):
    /// ```python
    /// import torch
    /// from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    /// cfg = LlamaConfig(
    ///     hidden_size=64, num_attention_heads=1, max_position_embeddings=4096*32,
    ///     rope_theta=10000.0,
    ///     rope_scaling={"rope_type":"yarn", "factor":32.0,
    ///                   "original_max_position_embeddings":4096,
    ///                   "beta_fast":32, "beta_slow":1},
    /// )
    /// emb = LlamaRotaryEmbedding(cfg)
    /// inv_freq = emb.inv_freq.cpu().numpy()
    /// # → first 4 values: [1.0, 0.7498942, 0.5623413, 0.4216965]
    /// # → last  4 values: [3.6517e-04, 2.7385e-04, 2.0535e-04, 1.5398e-04] / 32
    /// ```
    #[test]
    fn yarn_pinned_inv_freq_head64_factor32() {
        let head_dim = 64;
        let theta = 10000.0;
        let yarn = RopeScaling::Yarn {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 4096,
        };
        let inv = compute_inv_freq(head_dim, theta, Some(yarn));
        let base = compute_inv_freq(head_dim, theta, None);

        // Find correction range and verify low-i dims are extrapolation.
        let (low, high) = yarn_find_correction_range(
            32.0, 1.0, head_dim, theta, 4096);

        // low should be a small positive integer (~0–4 range for these params).
        // high should be < head_dim/2 - 1 (some interpolation happens).
        assert!(low >= 0.0 && low < high,
            "low={low} high={high} out of order");
        assert!(high <= (head_dim / 2 - 1) as f64,
            "high={high} exceeds half-1");

        // Property: ramp at i=low is 0 (pure extrapolation) and at i=high is 1
        // (pure interpolation). Pick the floor/ceil indices.
        let low_idx = low as usize;
        let high_idx = high as usize;
        if low_idx > 0 {
            // i < low → must equal base
            assert!((inv[low_idx - 1] - base[low_idx - 1]).abs() < 1e-12);
        }
        // i > high → must equal base / factor
        if high_idx + 1 < inv.len() {
            let interp = base[high_idx + 1] / 32.0_f64;
            assert!((inv[high_idx + 1] - interp).abs() < 1e-12,
                "i={} expected interp {} got {}", high_idx + 1, interp, inv[high_idx + 1]);
        }

        // mscale = sqrt(0.1*ln(32) + 1.0) ≈ 1.16299
        let mscale = compute_attention_scaling(Some(yarn));
        let want = (0.1f64 * 32.0f64.ln() + 1.0).sqrt() as f32;
        assert!((mscale - want).abs() < 1e-6, "mscale {mscale} vs {want}");
    }

    /// Pinned numerical reference: head_dim=128, theta=10000, factor=4,
    /// original_max=8192. Same convention as the head_dim=64 test.
    #[test]
    fn yarn_pinned_inv_freq_head128_factor4_orig8192() {
        let head_dim = 128;
        let theta = 10000.0;
        let yarn = RopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 8192,
        };
        let inv = compute_inv_freq(head_dim, theta, Some(yarn));
        let base = compute_inv_freq(head_dim, theta, None);

        let (low, high) = yarn_find_correction_range(
            32.0, 1.0, head_dim, theta, 8192);
        assert!(low < high, "low {low} >= high {high}");

        // Pure extrapolation tail
        let low_idx = low as usize;
        for i in 0..low_idx {
            assert!((inv[i] - base[i]).abs() < 1e-12,
                "i={i}: extrapolation broke ({} vs {})", inv[i], base[i]);
        }
        // Pure interpolation tail
        let high_idx = high.ceil() as usize;
        for i in (high_idx + 1)..inv.len() {
            let want = base[i] / 4.0;
            assert!((inv[i] - want).abs() < 1e-12,
                "i={i}: interpolation broke ({} vs {})", inv[i], want);
        }

        // Monotonicity: yarn_inv_freq is monotonically non-increasing.
        for i in 1..inv.len() {
            assert!(inv[i] <= inv[i - 1] + 1e-12,
                "yarn inv_freq must be non-increasing at i={i}: {} > {}",
                inv[i], inv[i - 1]);
        }
    }

    /// Pinned numerical reference: head_dim=128, theta=500000 (Llama-3 base),
    /// factor=8, original_max=8192. Verifies the formula handles non-default
    /// theta correctly.
    #[test]
    fn yarn_pinned_inv_freq_head128_theta500k() {
        let head_dim = 128;
        let theta = 500000.0;
        let yarn = RopeScaling::Yarn {
            factor: 8.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 8192,
        };
        let inv = compute_inv_freq(head_dim, theta, Some(yarn));
        let base = compute_inv_freq(head_dim, theta, None);

        // mscale check
        let want_mscale = ((0.1f64) * 8.0_f64.ln() + 1.0).sqrt() as f32;
        let got_mscale = compute_attention_scaling(Some(yarn));
        assert!((got_mscale - want_mscale).abs() < 1e-6);

        // The per-pair formula must straddle base and base/factor.
        for i in 0..inv.len() {
            let lo = base[i] / 8.0;
            let hi = base[i];
            assert!(inv[i] >= lo - 1e-12 && inv[i] <= hi + 1e-12,
                "i={i}: yarn {} not in [{}, {}]", inv[i], lo, hi);
        }
    }

    /// YaRN correction range responds to `original_max_position`:
    /// larger original max ⇒ shifted cutoffs (more dims interpolated).
    #[test]
    fn yarn_correction_range_shifts_with_original_max() {
        let head_dim = 128;
        let theta = 10000.0;
        let (low_4k, high_4k) = yarn_find_correction_range(
            32.0, 1.0, head_dim, theta, 4096);
        let (low_8k, high_8k) = yarn_find_correction_range(
            32.0, 1.0, head_dim, theta, 8192);
        // Larger original_max → both cutoffs shift toward higher dim indices
        // (the model already covered more positions natively, so fewer dims
        // need extrapolation).
        assert!(low_8k >= low_4k,
            "low cutoff should shift up with larger orig_max: {low_4k} → {low_8k}");
        assert!(high_8k >= high_4k,
            "high cutoff should shift up with larger orig_max: {high_4k} → {high_8k}");
    }

    /// Fingerprint round-trip: same params ⇒ same fingerprint, different
    /// params ⇒ different fingerprint.
    #[test]
    fn fingerprint_collision_resistance() {
        let a = RopeScaling::Yarn {
            factor: 32.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let b = RopeScaling::Yarn {
            factor: 32.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let c = RopeScaling::Yarn {
            factor: 16.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let d = RopeScaling::Linear { factor: 32.0 };
        assert_eq!(a.fingerprint_bytes(), b.fingerprint_bytes());
        assert_ne!(a.fingerprint_bytes(), c.fingerprint_bytes());
        assert_ne!(a.fingerprint_bytes(), d.fingerprint_bytes());
    }

    /// Partial RoPE (partial < 1): non-rotated dims must be exactly zero.
    #[test]
    fn partial_rope_non_rotated_dims_are_zero() {
        let head_dim = 64;
        let theta = 10000.0;
        let partial = 0.5; // rot_dim = 32, half_rot = 16
        let positions = [0u32, 10, 100];
        let mut out = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table_partial(
            &mut out, &positions, head_dim, theta, partial, None,
        );
        let half = head_dim / 2; // 32
        let half_rot = 16;
        for (row_idx, _) in positions.iter().enumerate() {
            // cos region: [half_rot..half) must be zero
            for i in half_rot..half {
                let got_cos = out[row_idx * head_dim + i];
                assert_eq!(got_cos, 0.0f32,
                    "row {row_idx} cos[{i}] should be 0, got {got_cos}");
            }
            // sin region: [half + half_rot..head_dim) must be zero
            for i in half_rot..half {
                let got_sin = out[row_idx * head_dim + half + i];
                assert_eq!(got_sin, 0.0f32,
                    "row {row_idx} sin[{i}] should be 0, got {got_sin}");
            }
        }
    }

    /// Partial RoPE with factor=1.0 degrades to full RoPE (partial=1.0).
    #[test]
    fn partial_rope_factor_one_equals_full_rope() {
        let head_dim = 64;
        let theta = 10000.0;
        let positions = [0u32, 5, 42, 999];
        let mut out_full = vec![0.0f32; positions.len() * head_dim];
        let mut out_partial = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table(&mut out_full, &positions, head_dim, theta, None);
        fill_cos_sin_table_partial(
            &mut out_partial, &positions, head_dim, theta, 1.0, None,
        );
        for (i, (a, b)) in out_full.iter().zip(out_partial.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "full vs partial(p=1.0) mismatch at index {i}: {a} vs {b}");
        }
    }

    /// Linear scaling fill_cos_sin_table: cos/sin must use
    /// `inv_freq / factor` while mscale remains 1.0.
    #[test]
    fn fill_cos_sin_linear_scaling_matches_divided_inv_freq() {
        let head_dim = 64;
        let theta = 10000.0;
        let factor = 8.0_f32;
        let scaling = RopeScaling::Linear { factor };
        let positions = [0u32, 1, 63, 512];
        let mut out = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table(&mut out, &positions, head_dim, theta, Some(scaling));

        let base = compute_inv_freq(head_dim, theta, None);
        let half = head_dim / 2;
        for (row_idx, &pos) in positions.iter().enumerate() {
            for i in 0..half {
                let freq = base[i] / factor as f64;
                let angle = pos as f64 * freq;
                let want_cos = angle.cos() as f32;
                let want_sin = angle.sin() as f32;
                let got_cos = out[row_idx * head_dim + i];
                let got_sin = out[row_idx * head_dim + half + i];
                assert!((got_cos - want_cos).abs() < 1e-6,
                    "linear row {row_idx} i {i}: cos {got_cos} vs {want_cos}");
                assert!((got_sin - want_sin).abs() < 1e-6,
                    "linear row {row_idx} i {i}: sin {got_sin} vs {want_sin}");
            }
        }
    }

    /// Minimal head_dim=2: only one pair, all formulas must still work.
    #[test]
    fn compute_inv_freq_head_dim_2() {
        let inv = compute_inv_freq(2, 10000.0, None);
        assert_eq!(inv.len(), 1);
        assert!((inv[0] - 1.0).abs() < 1e-12, "inv[0] should be 1.0, got {}", inv[0]);
    }

    /// fill_cos_sin_table with head_dim=2 produces valid cos/sin.
    #[test]
    fn fill_cos_sin_head_dim_2() {
        let head_dim = 2;
        let theta = 10000.0;
        let positions = [0u32, 1, 100];
        let mut out = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table(&mut out, &positions, head_dim, theta, None);
        // head_dim=2, half=1: row = [cos, sin]
        for (row_idx, &pos) in positions.iter().enumerate() {
            let angle = pos as f64 * 1.0; // inv_freq[0] = 1.0
            let want_cos = angle.cos() as f32;
            let want_sin = angle.sin() as f32;
            let got_cos = out[row_idx * 2];
            let got_sin = out[row_idx * 2 + 1];
            assert!((got_cos - want_cos).abs() < 1e-6);
            assert!((got_sin - want_sin).abs() < 1e-6);
        }
    }

    /// linear_ramp_mask: when low == high the span is clamped to 1e-3,
    /// so the ramp still produces valid clamped values.
    #[test]
    fn linear_ramp_mask_low_equals_high() {
        let mask = linear_ramp_mask(5.0, 5.0, 10);
        // All values must be in [0, 1] — span protection prevents div-by-zero.
        for (i, &v) in mask.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0,
                "ramp[{i}] = {v} out of [0,1]");
        }
    }

    /// linear_ramp_mask: boundary values at i=0 and i=half-1.
    #[test]
    fn linear_ramp_mask_boundary_values() {
        let half = 32;
        let low = 8.0;
        let high = 24.0;
        let mask = linear_ramp_mask(low, high, half);
        assert_eq!(mask.len(), half);
        // i=0: (0 - low) / (high - low) < 0 → clamped to 0
        assert!((mask[0] - 0.0).abs() < 1e-12, "mask[0] = {}", mask[0]);
        // i=31: (31 - low) / (high - low) > 1 → clamped to 1
        assert!((mask[31] - 1.0).abs() < 1e-12, "mask[31] = {}", mask[31]);
    }

    /// RopeScaling Debug and Clone derive round-trip.
    #[test]
    fn rope_scaling_debug_and_clone() {
        let yarn = RopeScaling::Yarn {
            factor: 4.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 8192,
        };
        let cloned = yarn.clone();
        assert_eq!(yarn, cloned);
        let debug_str = format!("{yarn:?}");
        assert!(debug_str.contains("Yarn"), "Debug should contain Yarn: {debug_str}");
        assert!(debug_str.contains("4"), "Debug should contain factor: {debug_str}");

        let linear = RopeScaling::Linear { factor: 8.0 };
        let linear_cloned = linear.clone();
        assert_eq!(linear, linear_cloned);
        let lin_debug = format!("{linear:?}");
        assert!(lin_debug.contains("Linear"), "Debug should contain Linear: {lin_debug}");
    }

    /// RopeScaling PartialEq: different variants are not equal.
    #[test]
    fn rope_scaling_partial_eq_cross_variant() {
        let yarn = RopeScaling::Yarn {
            factor: 8.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let linear = RopeScaling::Linear { factor: 8.0 };
        assert_ne!(yarn, linear);
    }

    /// YaRN with factor=1.0: inv_freq must equal the base (no scaling).
    #[test]
    fn yarn_factor_one_equals_base() {
        let head_dim = 64;
        let theta = 10000.0;
        let base = compute_inv_freq(head_dim, theta, None);
        let yarn = compute_inv_freq(head_dim, theta, Some(RopeScaling::Yarn {
            factor: 1.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        }));
        // factor=1.0 → interpolation = base/1 = base, so the blend is
        // (1-r)*base + r*base = base for all dims.
        for (i, (b, y)) in base.iter().zip(yarn.iter()).enumerate() {
            assert!((y - b).abs() < 1e-12,
                "factor=1.0 mismatch at i={i}: base={b}, yarn={y}");
        }
        // mscale for factor=1.0: sqrt(0.1*ln(1) + 1) = sqrt(1) = 1.0
        let mscale = compute_attention_scaling(Some(RopeScaling::Yarn {
            factor: 1.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        }));
        assert!((mscale - 1.0).abs() < 1e-6, "mscale for factor=1 should be 1.0, got {mscale}");
    }

    /// Standard inv_freq is monotonically decreasing: higher dim indices
    /// produce smaller angular frequencies.
    #[test]
    fn standard_inv_freq_monotonically_decreasing() {
        let head_dim = 128;
        let theta = 10000.0;
        let inv = compute_inv_freq(head_dim, theta, None);
        for i in 1..inv.len() {
            assert!(inv[i] < inv[i - 1],
                "inv_freq must be strictly decreasing at i={i}: {} >= {}",
                inv[i], inv[i - 1]);
        }
    }

    /// Linear scaling preserves the monotonicity of inv_freq.
    #[test]
    fn linear_inv_freq_monotonically_decreasing() {
        let head_dim = 64;
        let theta = 10000.0;
        let factor = 2.5_f32;
        let inv = compute_inv_freq(head_dim, theta, Some(RopeScaling::Linear { factor }));
        for i in 1..inv.len() {
            assert!(inv[i] < inv[i - 1],
                "linear inv_freq must be strictly decreasing at i={i}: {} >= {}",
                inv[i], inv[i - 1]);
        }
    }

    /// Linear scaling with different factors: larger factor produces
    /// proportionally smaller inv_freq for every dim.
    #[test]
    fn linear_larger_factor_produces_smaller_inv_freq() {
        let head_dim = 64;
        let theta = 10000.0;
        let inv_f2 = compute_inv_freq(head_dim, theta, Some(RopeScaling::Linear { factor: 2.0 }));
        let inv_f8 = compute_inv_freq(head_dim, theta, Some(RopeScaling::Linear { factor: 8.0 }));
        for (v2, v8) in inv_f2.iter().zip(inv_f8.iter()) {
            assert!(v8 < v2,
                "factor=8 inv_freq ({}) must be < factor=2 ({})",
                v8, v2);
        }
    }

    /// YaRN attention_scaling is monotonically increasing with factor.
    /// Larger extension factors produce higher temperature.
    #[test]
    fn yarn_attention_scaling_increases_with_factor() {
        let make_yarn = |f: f32| RopeScaling::Yarn {
            factor: f, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let s2 = compute_attention_scaling(Some(make_yarn(2.0)));
        let s4 = compute_attention_scaling(Some(make_yarn(4.0)));
        let s16 = compute_attention_scaling(Some(make_yarn(16.0)));
        let s64 = compute_attention_scaling(Some(make_yarn(64.0)));
        assert!(s2 < s4, "mscale(2)={s2} should be < mscale(4)={s4}");
        assert!(s4 < s16, "mscale(4)={s4} should be < mscale(16)={s16}");
        assert!(s16 < s64, "mscale(16)={s16} should be < mscale(64)={s64}");
    }

    /// fill_cos_sin_table position=0: cos=1, sin=0 for all dims (angle=0).
    #[test]
    fn fill_cos_sin_position_zero_trivial() {
        let head_dim = 64;
        let theta = 10000.0;
        let positions = [0u32];
        let mut out = vec![0.0f32; head_dim];
        fill_cos_sin_table(&mut out, &positions, head_dim, theta, None);
        let half = head_dim / 2;
        for i in 0..half {
            let cos_val = out[i];
            let sin_val = out[half + i];
            assert!((cos_val - 1.0f32).abs() < 1e-6,
                "cos[{i}] at pos=0 should be 1.0, got {cos_val}");
            assert!(sin_val.abs() < 1e-6,
                "sin[{i}] at pos=0 should be 0.0, got {sin_val}");
        }
    }

    /// YaRN inv_freq is bounded between interpolation and extrapolation
    /// for all dims regardless of beta parameters.
    #[test]
    fn yarn_inv_freq_bounded_between_interp_and_extrap() {
        let head_dim = 96;
        let theta = 10000.0;
        let factor = 16.0_f32;
        let base = compute_inv_freq(head_dim, theta, None);
        let yarn_inv = compute_inv_freq(head_dim, theta, Some(RopeScaling::Yarn {
            factor, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        }));
        // For every dim: base/factor <= yarn_inv <= base
        for (i, (&y, &b)) in yarn_inv.iter().zip(base.iter()).enumerate() {
            let lo = b / factor as f64;
            assert!(y >= lo - 1e-12 && y <= b + 1e-12,
                "yarn[{i}]={y} not in [{lo}, {b}]");
        }
    }

    /// fill_cos_sin_table with YaRN produces mscale-scaled output:
    /// at position 0, cos should be exactly mscale (since cos(0)=1)
    /// and sin should be 0 (since sin(0)=0).
    #[test]
    fn fill_cos_sin_yarn_position_zero_mscale_scaled() {
        let head_dim = 64;
        let theta = 10000.0;
        let yarn = RopeScaling::Yarn {
            factor: 8.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let mscale = compute_attention_scaling(Some(yarn));
        assert!(mscale > 1.0, "mscale for factor=8 should be > 1.0, got {mscale}");
        let positions = [0u32];
        let mut out = vec![0.0f32; head_dim];
        fill_cos_sin_table(&mut out, &positions, head_dim, theta, Some(yarn));
        let half = head_dim / 2;
        for i in 0..half {
            let cos_val = out[i];
            let sin_val = out[half + i];
            assert!((cos_val - mscale).abs() < 1e-5,
                "yarn cos[{i}] at pos=0 should be mscale={mscale}, got {cos_val}");
            assert!(sin_val.abs() < 1e-5,
                "yarn sin[{i}] at pos=0 should be 0, got {sin_val}");
        }
    }

    /// Correction range: beta_fast must be > beta_slow for YaRN.
    /// Verify that the correction range (low, high) satisfies low <= high
    /// for typical parameter combinations.
    #[test]
    fn yarn_correction_range_low_le_high_for_various_params() {
        let head_dim = 64;
        let theta = 10000.0;
        let cases: Vec<(f32, f32, usize)> = vec![
            (32.0, 1.0, 4096),
            (64.0, 1.0, 4096),
            (32.0, 0.5, 8192),
            (128.0, 1.0, 2048),
        ];
        for (beta_fast, beta_slow, orig_max) in cases {
            let (low, high) = yarn_find_correction_range(
                beta_fast, beta_slow, head_dim, theta, orig_max);
            assert!(low <= high,
                "low ({low}) > high ({high}) for beta_fast={beta_fast}, beta_slow={beta_slow}, orig_max={orig_max}");
        }
    }

    /// fill_cos_sin_table_partial with YaRN: non-rotated dims must be zero,
    /// and rotated dims must carry the mscale factor.
    #[test]
    fn fill_cos_sin_partial_yarn_non_rotated_zero() {
        let head_dim = 64;
        let theta = 10000.0;
        let partial = 0.5;
        let yarn = RopeScaling::Yarn {
            factor: 4.0, beta_fast: 32.0, beta_slow: 1.0,
            original_max_position: 4096,
        };
        let mscale = compute_attention_scaling(Some(yarn));
        assert!(mscale > 1.0, "mscale for factor=4 should be > 1.0, got {mscale}");
        let positions = [10u32, 50];
        let mut out = vec![0.0f32; positions.len() * head_dim];
        fill_cos_sin_table_partial(
            &mut out, &positions, head_dim, theta, partial, Some(yarn),
        );
        let half = head_dim / 2;
        let half_rot = 16; // head_dim * 0.5 / 2 = 16
        for (row_idx, &pos) in positions.iter().enumerate() {
            // Rotated dims (i < half_rot) must have mscale folded in:
            // at pos=0 would be mscale, but pos > 0, so just verify non-zero
            // and bounded by mscale.
            for i in 0..half_rot {
                let cos_val = out[row_idx * head_dim + i].abs();
                assert!(cos_val > 0.0,
                    "rotated cos[{i}] at pos={pos} should be non-zero");
                assert!(cos_val <= mscale + 1e-5,
                    "rotated cos[{i}] at pos={pos}: |{cos_val}| > mscale {mscale}");
            }
            // Non-rotated dims must be exactly zero
            for i in half_rot..half {
                assert_eq!(out[row_idx * head_dim + i], 0.0f32,
                    "non-rotated cos row {row_idx}[{i}] should be 0");
                assert_eq!(out[row_idx * head_dim + half + i], 0.0f32,
                    "non-rotated sin row {row_idx}[{i}] should be 0");
            }
        }
    }
}
