//! Scalar RoPE (Rotary Position Embedding).

/// RoPE: non-interleaved rotary position embedding.
///
/// For each head, pairs `(x[2i], x[2i+1])` are rotated by `(cos[i], sin[i])`:
/// ```text
/// out[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
/// out[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]
/// ```
///
/// Layout: `x` and `out` are `[n_heads, head_dim]`, `cos`/`sin` are `[head_dim/2]`.
#[no_mangle]
#[inline(never)]
pub extern "C" fn scalar_rope(
    x: *const f32,
    cos: *const f32,
    sin: *const f32,
    out: *mut f32,
    head_dim: usize,
    n_heads: usize,
) {
    let half = head_dim / 2;
    unsafe {
        for h in 0..n_heads {
            let base = h * head_dim;
            for i in 0..half {
                let x0 = *x.add(base + 2 * i);
                let x1 = *x.add(base + 2 * i + 1);
                let c = *cos.add(i);
                let s = *sin.add(i);
                *out.add(base + 2 * i) = x0 * c - x1 * s;
                *out.add(base + 2 * i + 1) = x1 * c + x0 * s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_ops_rope_identity() {
        // cos=1, sin=0 → identity
        let head_dim = 4;
        let n_heads = 2;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cos = vec![1.0_f32; head_dim / 2];
        let sin = vec![0.0_f32; head_dim / 2];
        let mut out = vec![0.0_f32; n_heads * head_dim];

        scalar_rope(
            x.as_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            out.as_mut_ptr(),
            head_dim,
            n_heads,
        );

        assert_eq!(out, x);
    }

    #[test]
    fn test_scalar_ops_rope_90_degrees() {
        // cos=0, sin=1 → 90-degree rotation: (x0,x1) → (-x1, x0)
        let head_dim = 4;
        let n_heads = 1;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let cos = vec![0.0_f32; head_dim / 2];
        let sin = vec![1.0_f32; head_dim / 2];
        let mut out = vec![0.0_f32; head_dim];

        scalar_rope(
            x.as_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            out.as_mut_ptr(),
            head_dim,
            n_heads,
        );

        // out[0] = 1*0 - 2*1 = -2, out[1] = 2*0 + 1*1 = 1
        // out[2] = 3*0 - 4*1 = -4, out[3] = 4*0 + 3*1 = 3
        assert_eq!(out, vec![-2.0, 1.0, -4.0, 3.0]);
    }

    #[test]
    fn test_scalar_ops_rope_preserves_norm() {
        // Rotation preserves vector norm
        let head_dim = 4;
        let n_heads = 1;
        let x = vec![3.0_f32, 4.0, 1.0, 2.0];
        let angle = 0.5_f32;
        let cos = vec![angle.cos(); head_dim / 2];
        let sin = vec![angle.sin(); head_dim / 2];
        let mut out = vec![0.0_f32; head_dim];

        scalar_rope(
            x.as_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            out.as_mut_ptr(),
            head_dim,
            n_heads,
        );

        let norm_in: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_out: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_in - norm_out).abs() < 1e-5,
            "norm changed: {norm_in} -> {norm_out}"
        );
    }
}
