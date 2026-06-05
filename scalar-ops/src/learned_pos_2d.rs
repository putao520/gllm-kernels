//! SigLIP / ViT 2D learned positional embedding scalar reference (T44).
//!
//! ```text
//! out[p, d] = patches[p, d] + pos_table[p, d]
//! ```
//!
//! 布局 (均为 row-major `[num_patches, embed_dim]`):
//! - `patches`   : PatchEmbed 输出
//! - `pos_table` : 学习的位置嵌入表
//! - `out`       : 位置编码后的 patch 序列
//!
//! 本函数仅作为 Phase 0 SymExec trace 提取目标 + 数值 ground truth (测试用),
//! **禁止在运行时/测试代码中直接调用** (违反 CLAUDE.md NO_SCALAR 铁律)。

/// Learned 2D positional embedding scalar reference (pure elementwise add).
///
/// # Safety
///
/// - `patches` / `pos_table` / `out` 必须指向 `num_patches * embed_dim` 个 f32。
/// - 所有指针互不 alias。
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_learned_pos_2d(
    patches: *const f32,
    pos_table: *const f32,
    out: *mut f32,
    num_patches: usize,
    embed_dim: usize,
) {
    let n = num_patches * embed_dim;
    for i in 0..n {
        *out.add(i) = *patches.add(i) + *pos_table.add(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_pos_2d_elementwise_add_small() {
        let np = 4;
        let ed = 3;
        let patches: Vec<f32> = (0..np * ed).map(|i| i as f32).collect();
        let pos: Vec<f32> = (0..np * ed).map(|i| (i as f32) * 0.5).collect();
        let mut out = vec![0.0_f32; np * ed];
        unsafe {
            scalar_learned_pos_2d(patches.as_ptr(), pos.as_ptr(), out.as_mut_ptr(), np, ed);
        }
        for i in 0..np * ed {
            let expected = patches[i] + pos[i];
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "idx {i}: got {} expected {}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn learned_pos_2d_zero_pos_is_identity() {
        let np = 2;
        let ed = 5;
        let patches: Vec<f32> = (0..np * ed).map(|i| (i as f32) + 0.125).collect();
        let pos = vec![0.0_f32; np * ed];
        let mut out = vec![0.0_f32; np * ed];
        unsafe {
            scalar_learned_pos_2d(patches.as_ptr(), pos.as_ptr(), out.as_mut_ptr(), np, ed);
        }
        for i in 0..np * ed {
            assert!((out[i] - patches[i]).abs() < 1e-6);
        }
    }
}
