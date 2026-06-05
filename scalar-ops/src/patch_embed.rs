//! Scalar Patch Embedding (SigLIP / ViT vision tower 核心 op, T44).
//!
//! ViT 风格的 patch embedding: 对图像做 Conv2D (kernel_size = patch_size,
//! stride = patch_size) 得到 `num_patches = (image_size / patch_size)^2` 个
//! patch token, 每个 token 的维度为 `embed_dim`。
//!
//! 布局 (均为 row-major):
//! - `image`   : `[in_channels, image_size, image_size]`
//! - `kernel`  : `[embed_dim, in_channels, patch_size, patch_size]`
//! - `patches` : `[num_patches, embed_dim]`,
//!   其中 `num_patches = (image_size / patch_size)^2`, 按 row-major 扁平化
//!   (先行后列: `p = p_row * num_patches_side + p_col`)。
//!
//! 数学定义:
//! ```text
//! patches[p, e] = Σ_{c, kr, kc}
//!   image[c, p_row*patch_size + kr, p_col*patch_size + kc]
//!   * kernel[e, c, kr, kc]
//! ```
//!
//! 本函数仅作为 Phase 0 SymExec trace 提取目标 + 数值 ground truth (测试用),
//! **禁止在运行时/测试代码中直接调用** (违反 CLAUDE.md NO_SCALAR 铁律)。

/// Patch embedding via Conv2D sliding window (stride = patch_size, no padding).
///
/// # Safety
///
/// - `image`   必须指向 `in_channels * image_size * image_size` 个 f32。
/// - `kernel`  必须指向 `embed_dim * in_channels * patch_size * patch_size` 个 f32。
/// - `patches` 必须指向 `num_patches * embed_dim` 个 f32,
///   `num_patches = (image_size / patch_size) ^ 2`。
/// - 所有指针互不 alias。
/// - `image_size % patch_size == 0` (ViT 标配, 非整除在 ViT 语义未定义)。
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_patch_embed(
    image: *const f32,
    kernel: *const f32,
    patches: *mut f32,
    patch_size: usize,
    embed_dim: usize,
    in_channels: usize,
    image_size: usize,
) {
    let num_patches_side = image_size / patch_size;
    let image_plane = image_size * image_size; // elements per input channel
    let kernel_plane = patch_size * patch_size; // elements per (embed, channel)

    for p_row in 0..num_patches_side {
        for p_col in 0..num_patches_side {
            let p = p_row * num_patches_side + p_col;
            for e in 0..embed_dim {
                let mut acc: f32 = 0.0;
                for c in 0..in_channels {
                    for kr in 0..patch_size {
                        for kc in 0..patch_size {
                            // image[c, p_row*patch_size + kr, p_col*patch_size + kc]
                            let img_row = p_row * patch_size + kr;
                            let img_col = p_col * patch_size + kc;
                            let img_idx =
                                c * image_plane + img_row * image_size + img_col;
                            // kernel[e, c, kr, kc]
                            let ker_idx =
                                e * in_channels * kernel_plane + c * kernel_plane + kr * patch_size + kc;
                            acc += *image.add(img_idx) * *kernel.add(ker_idx);
                        }
                    }
                }
                *patches.add(p * embed_dim + e) = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 小用例手算:
    ///   in_channels = 1, image_size = 2, patch_size = 2, embed_dim = 1
    ///   num_patches = 1 (整张图就是一个 patch)
    ///   image  = [[1, 2], [3, 4]]  (1×2×2)
    ///   kernel = [[1, 2], [3, 4]]  (1×1×2×2)
    ///   patches[0, 0] = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    #[test]
    fn patch_embed_single_patch_hand_computed() {
        let patch_size = 2;
        let embed_dim = 1;
        let in_channels = 1;
        let image_size = 2;

        let image = vec![1.0_f32, 2.0, 3.0, 4.0];
        let kernel = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut patches = vec![0.0_f32; 1];

        unsafe {
            scalar_patch_embed(
                image.as_ptr(),
                kernel.as_ptr(),
                patches.as_mut_ptr(),
                patch_size,
                embed_dim,
                in_channels,
                image_size,
            );
        }

        // 1 + 4 + 9 + 16 = 30
        assert!((patches[0] - 30.0).abs() < 1e-6, "got {}", patches[0]);
    }

    /// 多 patch 用例: kernel = 全 1, embed_dim = 1, in_channels = 1,
    /// image 每个格子递增, 手算每个 patch = 该 2×2 区块之和。
    ///   image_size = 4, patch_size = 2 → 4 个 patch (2×2 网格)
    ///   image = 1..=16 (row-major):
    ///     [ 1,  2,  3,  4]
    ///     [ 5,  6,  7,  8]
    ///     [ 9, 10, 11, 12]
    ///     [13, 14, 15, 16]
    ///   patches:
    ///     p=0 (row=0,col=0): 1+2+5+6     = 14
    ///     p=1 (row=0,col=1): 3+4+7+8     = 22
    ///     p=2 (row=1,col=0): 9+10+13+14  = 46
    ///     p=3 (row=1,col=1): 11+12+15+16 = 54
    #[test]
    fn patch_embed_kernel_sum_windowed_ones() {
        let patch_size = 2;
        let embed_dim = 1;
        let in_channels = 1;
        let image_size = 4;

        let image: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let kernel = vec![1.0_f32; embed_dim * in_channels * patch_size * patch_size];
        let num_patches = (image_size / patch_size).pow(2);
        let mut patches = vec![0.0_f32; num_patches * embed_dim];

        unsafe {
            scalar_patch_embed(
                image.as_ptr(),
                kernel.as_ptr(),
                patches.as_mut_ptr(),
                patch_size,
                embed_dim,
                in_channels,
                image_size,
            );
        }

        let expected = [14.0_f32, 22.0, 46.0, 54.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (patches[i] - e).abs() < 1e-5,
                "patch {i}: got {}, expected {e}",
                patches[i]
            );
        }
    }
}
