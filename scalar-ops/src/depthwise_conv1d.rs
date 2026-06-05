//! Scalar Depthwise 1D Convolution (USM Conformer convolution module 核心 op).
//!
//! 每个 channel 独立 1D 卷积, 输入 `[seq_len, channels]` + 权重 `[channels, kernel_size]`
//! → 输出 `[seq_len, channels]`。`causal = true` 时在 seq 维度前 zero-pad
//! `kernel_size - 1` 个元素, 保证 `output[t, c]` 只依赖 `input[0..=t, c]`
//! (Conformer 推理标准配置)。
//!
//! 数学定义 (row-major, causal = true):
//! ```text
//! output[t, c] = Σ_{k=0..kernel_size} input[t - (kernel_size - 1) + k, c] * weight[c, k]
//! ```
//! 其中 `input[i, c] = 0` for `i < 0` (zero padding).
//!
//! `causal = false` 时为 "SAME" padding, 在前后各 pad `(kernel_size - 1) / 2`
//! (仅支持奇数 kernel_size, 偶数情况按 causal 对称未定义)。

/// Depthwise 1D convolution (per-channel).
///
/// Inputs:
/// - `x`:      `[seq_len * channels]` row-major, `x[t * channels + c]`
/// - `weight`: `[channels * kernel_size]` row-major, `weight[c * kernel_size + k]`
/// - `out`:    `[seq_len * channels]` row-major (same layout as x)
///
/// `causal` 见模块文档。
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_depthwise_conv1d(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
    causal: u32,
) {
    let causal_mode = causal != 0;
    // causal: pad = kernel_size - 1 个元素放在序列开头 (左侧 zero pad)
    // non-causal ("SAME"): 前后各 pad (kernel_size - 1) / 2
    // t_input_origin 为输出位置 t 对应的"窗口左端在 input 坐标系中的偏移"
    //   causal:     t_input_origin = t - (kernel_size - 1)
    //   non-causal: t_input_origin = t - (kernel_size - 1) / 2
    let left_pad: isize = if causal_mode {
        (kernel_size as isize) - 1
    } else {
        ((kernel_size as isize) - 1) / 2
    };

    unsafe {
        for t in 0..seq_len {
            for c in 0..channels {
                let mut acc = 0.0_f32;
                for k in 0..kernel_size {
                    // window 内 k 对应的 input 时间索引
                    let t_in_signed = (t as isize) + (k as isize) - left_pad;
                    if t_in_signed < 0 || t_in_signed >= (seq_len as isize) {
                        // zero padding
                        continue;
                    }
                    let t_in = t_in_signed as usize;
                    let x_val = *x.add(t_in * channels + c);
                    let w_val = *weight.add(c * kernel_size + k);
                    acc += x_val * w_val;
                }
                *out.add(t * channels + c) = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手算 causal 卷积: seq=8, channels=2, kernel=3, causal=true。
    /// 权重 weight[c, k] = c * 10 + k + 1:
    ///   channel 0: w = [1, 2, 3]
    ///   channel 1: w = [11, 12, 13]
    /// 输入 x[t, c] = t + 1 (两通道同值):
    ///   x = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8]]
    ///
    /// Causal pad = 2 个 zero 在 t<0 处:
    ///   output[t, c] = x[t-2,c]*w[c,0] + x[t-1,c]*w[c,1] + x[t,c]*w[c,2]
    ///   (x[i<0, c] = 0)
    #[test]
    fn test_scalar_depthwise_conv1d_causal_basic() {
        let seq_len = 8;
        let channels = 2;
        let kernel = 3;

        let x: Vec<f32> = (0..seq_len)
            .flat_map(|t| (0..channels).map(move |_| (t + 1) as f32))
            .collect();
        let weight: Vec<f32> = (0..channels)
            .flat_map(|c| (0..kernel).map(move |k| (c * 10 + k + 1) as f32))
            .collect();
        let mut out = vec![0.0_f32; seq_len * channels];

        scalar_depthwise_conv1d(
            x.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
            seq_len,
            channels,
            kernel,
            1, // causal
        );

        // 手算 expected:
        //   channel 0 (w=[1,2,3]):
        //     t=0: 0*1 + 0*2 + 1*3 = 3
        //     t=1: 0*1 + 1*2 + 2*3 = 8
        //     t=2: 1*1 + 2*2 + 3*3 = 14
        //     t=3: 2*1 + 3*2 + 4*3 = 20
        //     t=4: 3*1 + 4*2 + 5*3 = 26
        //     t=5: 4*1 + 5*2 + 6*3 = 32
        //     t=6: 5*1 + 6*2 + 7*3 = 38
        //     t=7: 6*1 + 7*2 + 8*3 = 44
        //   channel 1 (w=[11,12,13]):
        //     t=0: 0*11 + 0*12 + 1*13 = 13
        //     t=1: 0*11 + 1*12 + 2*13 = 38
        //     t=2: 1*11 + 2*12 + 3*13 = 74
        //     t=3: 2*11 + 3*12 + 4*13 = 110
        //     t=4: 3*11 + 4*12 + 5*13 = 146
        //     t=5: 4*11 + 5*12 + 6*13 = 182
        //     t=6: 5*11 + 6*12 + 7*13 = 218
        //     t=7: 6*11 + 7*12 + 8*13 = 254
        let expected_c0 = [3.0, 8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0];
        let expected_c1 = [13.0, 38.0, 74.0, 110.0, 146.0, 182.0, 218.0, 254.0];

        for t in 0..seq_len {
            let got_c0 = out[t * channels + 0];
            let got_c1 = out[t * channels + 1];
            assert!(
                (got_c0 - expected_c0[t]).abs() < 1e-4,
                "channel0 t={t}: got {got_c0}, expected {}",
                expected_c0[t]
            );
            assert!(
                (got_c1 - expected_c1[t]).abs() < 1e-4,
                "channel1 t={t}: got {got_c1}, expected {}",
                expected_c1[t]
            );
        }
    }

    /// kernel_size = 1 退化为 per-channel scale: out[t, c] = x[t, c] * w[c, 0]。
    #[test]
    fn test_scalar_depthwise_conv1d_kernel_one_is_scale() {
        let seq_len = 4;
        let channels = 3;
        let kernel = 1;

        let x: Vec<f32> = (0..seq_len * channels).map(|i| (i + 1) as f32).collect();
        // w = [0.5, 1.0, 2.0] (一个 per-channel scale)
        let weight = vec![0.5_f32, 1.0, 2.0];
        let mut out = vec![0.0_f32; seq_len * channels];

        scalar_depthwise_conv1d(
            x.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
            seq_len,
            channels,
            kernel,
            1,
        );

        for t in 0..seq_len {
            for c in 0..channels {
                let got = out[t * channels + c];
                let expected = x[t * channels + c] * weight[c];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "t={t} c={c}: got {got}, expected {expected}"
                );
            }
        }
    }

    /// Non-causal (symmetric SAME) pad: kernel=3 → 前后各 pad 1 个零。
    /// out[t, c] = x[t-1, c]*w[0] + x[t, c]*w[1] + x[t+1, c]*w[2]
    #[test]
    fn test_scalar_depthwise_conv1d_noncausal_same_pad() {
        let seq_len = 5;
        let channels = 1;
        let kernel = 3;

        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0_f32, 1.0, 1.0]; // 均匀平均窗口
        let mut out = vec![0.0_f32; seq_len];

        scalar_depthwise_conv1d(
            x.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
            seq_len,
            channels,
            kernel,
            0, // non-causal
        );

        // 手算: left_pad = 1
        //   t=0: 0 + 1 + 2 = 3
        //   t=1: 1 + 2 + 3 = 6
        //   t=2: 2 + 3 + 4 = 9
        //   t=3: 3 + 4 + 5 = 12
        //   t=4: 4 + 5 + 0 = 9
        let expected = [3.0, 6.0, 9.0, 12.0, 9.0];
        for (t, &e) in expected.iter().enumerate() {
            assert!(
                (out[t] - e).abs() < 1e-5,
                "t={t}: got {}, expected {e}",
                out[t]
            );
        }
    }

    /// Zero input → zero output (纯安全性/初始化测试)。
    #[test]
    fn test_scalar_depthwise_conv1d_zero_input() {
        let seq_len = 6;
        let channels = 4;
        let kernel = 5;

        let x = vec![0.0_f32; seq_len * channels];
        let weight: Vec<f32> = (0..channels * kernel).map(|i| (i + 1) as f32).collect();
        let mut out = vec![1.0_f32; seq_len * channels]; // 预置非零，验证会被覆盖

        scalar_depthwise_conv1d(
            x.as_ptr(),
            weight.as_ptr(),
            out.as_mut_ptr(),
            seq_len,
            channels,
            kernel,
            1,
        );

        for &v in &out {
            assert!(v.abs() < 1e-6, "zero-input should give zero output, got {v}");
        }
    }
}
