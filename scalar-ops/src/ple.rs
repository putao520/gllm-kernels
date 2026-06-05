//! Per-Layer Embedding (PLE) scalar reference — Gemma 4 E2B/E4B.
//!
//! 计算公式 (见 SPEC/DOCS/architecture/gemma4-op-audit.md §1.5):
//! ```text
//! ple_ctx    = main_embed @ proj_w                         // [seq_len, dim]
//! signal     = (ple_ctx + ple_slice × √dim) / √2            // [seq_len, dim]
//! post_mlp   = signal @ post_mlp_w                          // [seq_len, hidden]
//! hidden_out = hidden + post_mlp                            // [seq_len, hidden]
//! ```
//!
//! 输入张量:
//! - `hidden`      [seq_len, hidden]         — 当前 hidden state
//! - `main_embed`  [seq_len, hidden]         — 主 token embedding
//! - `ple_slice`   [seq_len, dim]            — 当前层的 PLE token slice (已按 layer_idx 切出)
//! - `proj_w`      [hidden, dim]             — context-aware 投影权重
//! - `post_mlp_w`  [dim, hidden]             — 残差注入投影权重
//!
//! 输出: `out` [seq_len, hidden]
//!
//! 本函数仅作为 Phase 0 SymExec trace 提取目标 + 数值 ground truth (测试用),
//! **禁止在运行时/测试代码中直接调用** (违反 CLAUDE.md NO_SCALAR 铁律)。

/// Per-Layer Embedding scalar reference.
///
/// # Safety
///
/// - `hidden` / `main_embed` / `out` 必须指向 `seq_len * hidden_dim` 个 f32。
/// - `ple_slice` 必须指向 `seq_len * dim` 个 f32。
/// - `proj_w` 必须指向 `hidden_dim * dim` 个 f32 (row-major `[hidden, dim]`)。
/// - `post_mlp_w` 必须指向 `dim * hidden_dim` 个 f32 (row-major `[dim, hidden]`)。
/// - 所有指针互不 alias。
#[no_mangle]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn scalar_per_layer_embed(
    hidden: *const f32,
    main_embed: *const f32,
    ple_slice: *const f32,
    proj_w: *const f32,
    post_mlp_w: *const f32,
    out: *mut f32,
    seq_len: usize,
    hidden_dim: usize,
    dim: usize,
) {
    let sqrt_dim = (dim as f32).sqrt();
    let inv_sqrt_2 = 1.0_f32 / std::f32::consts::SQRT_2;

    // ple_ctx: [seq_len, dim] = main_embed @ proj_w
    // signal:  [seq_len, dim] = (ple_ctx + ple_slice * sqrt_dim) * inv_sqrt_2
    // post_mlp:[seq_len, hidden_dim] = signal @ post_mlp_w
    // out:     [seq_len, hidden_dim] = hidden + post_mlp

    for s in 0..seq_len {
        // 1) ple_ctx[s, j] = Σ_k main_embed[s, k] * proj_w[k, j]
        //    signal[s, j]  = (ple_ctx[s, j] + ple_slice[s, j] * sqrt_dim) * inv_sqrt_2
        //    先算 signal 并驻留在栈上; 后续 post_mlp 直接读取
        //    由于 dim 可能较大, 用堆分配
        let mut signal = vec![0.0_f32; dim];
        for j in 0..dim {
            let mut ctx = 0.0_f32;
            for k in 0..hidden_dim {
                ctx += *main_embed.add(s * hidden_dim + k) * *proj_w.add(k * dim + j);
            }
            let token = *ple_slice.add(s * dim + j) * sqrt_dim;
            signal[j] = (ctx + token) * inv_sqrt_2;
        }
        // 2) post_mlp[s, h] = Σ_j signal[j] * post_mlp_w[j, h]
        //    out[s, h] = hidden[s, h] + post_mlp[s, h]
        for h in 0..hidden_dim {
            let mut acc = 0.0_f32;
            for j in 0..dim {
                acc += signal[j] * *post_mlp_w.add(j * hidden_dim + h);
            }
            *out.add(s * hidden_dim + h) = *hidden.add(s * hidden_dim + h) + acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ple_numerical_identity_small() {
        // 极小用例: seq_len=1, hidden=2, dim=2
        // proj_w = I, post_mlp_w = I → signal = (main_embed + ple_slice*√2)/√2
        // out = hidden + signal (broadcast 到 hidden 维度, 因 dim==hidden)
        let hidden_buf = vec![1.0_f32, 2.0];
        let main_embed = vec![3.0_f32, 4.0];
        let ple_slice = vec![0.5_f32, 0.25];
        let proj_w = vec![1.0_f32, 0.0, 0.0, 1.0]; // I
        let post_mlp_w = vec![1.0_f32, 0.0, 0.0, 1.0]; // I
        let mut out = vec![0.0_f32; 2];
        unsafe {
            scalar_per_layer_embed(
                hidden_buf.as_ptr(),
                main_embed.as_ptr(),
                ple_slice.as_ptr(),
                proj_w.as_ptr(),
                post_mlp_w.as_ptr(),
                out.as_mut_ptr(),
                1,
                2,
                2,
            );
        }
        // signal[j] = (main_embed[j] + ple_slice[j]*√2) / √2
        //           = main_embed[j]/√2 + ple_slice[j]
        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        let s0 = 3.0 * inv_sqrt2 + 0.5;
        let s1 = 4.0 * inv_sqrt2 + 0.25;
        // out[h] = hidden[h] + signal[h]  (post_mlp_w = I)
        assert!((out[0] - (1.0 + s0)).abs() < 1e-5);
        assert!((out[1] - (2.0 + s1)).abs() < 1e-5);
    }

    #[test]
    fn ple_zero_weights_identity_residual() {
        // proj_w = 0, post_mlp_w = 0 → signal = 0 → out = hidden
        let seq = 2;
        let h = 3;
        let d = 2;
        let hidden_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let main_embed = vec![7.0; seq * h];
        let ple_slice = vec![0.5; seq * d];
        let proj_w = vec![0.0; h * d];
        let post_mlp_w = vec![0.0; d * h];
        let mut out = vec![0.0_f32; seq * h];
        unsafe {
            scalar_per_layer_embed(
                hidden_buf.as_ptr(),
                main_embed.as_ptr(),
                ple_slice.as_ptr(),
                proj_w.as_ptr(),
                post_mlp_w.as_ptr(),
                out.as_mut_ptr(),
                seq,
                h,
                d,
            );
        }
        for i in 0..seq * h {
            assert!((out[i] - hidden_buf[i]).abs() < 1e-6);
        }
    }
}
