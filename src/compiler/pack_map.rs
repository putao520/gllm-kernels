//! §0.2.7 虚拟权重 — PackMap 索引映射替代物理 pack (SSOT: §3.5)
//!
//! PackMap 描述 GEMM 权重矩阵的 pack 布局映射函数。
//! 当前: 运行时调用 pack_b() 将 RowMajor 权重重新排列为 PanelPacked 格式
//! 目标: 编译时生成 PackMap stride 计算 VmInstr，运行时零额外 pack buffer
//!
//! PackMap 是 (RowMajor × MapFn) → PhysicalOffset 的编译时映射：
//! - Identity: 无 pack，RowMajor 直读
//! - PanelPack: BLIS 风格 panel layout (mr×kc blocks)
//! - VnniPack: VNNI 4元素交织
//! - TilePack: AMX/GPU tile layout

/// §0.2.7 权重 pack 布局映射函数
#[derive(Debug, Clone, PartialEq)]
pub enum PackMap {
    /// 无 pack — 权重保持原始 RowMajor 布局，直读
    Identity,

    /// BLIS panel layout: B 矩阵按 nr×kc panel 存储
    /// 原始: B[k][n] (RowMajor, ldc=N)
    /// Panel: panel[j/nr][i/kc][r][c] = B[i*nr+r][j*kc+c]
    /// 映射: offset = panel_j * nr * kc + intra_panel_r * kc + intra_panel_c
    ///      = (col / nr) * (nr * kc) + (row % nr) * kc + (col % nr)
    ///      简化: = panel_id * nr * kc + r * kc + c
    /// 编译时可计算 stride，运行时零 pack 开销
    PanelPack {
        /// 微核列数 (x86 AVX-512: 4/32, AVX2: 3/16, NEON: 4/12)
        nr: usize,
        /// 分块深度 (来自 L1 缓存约束)
        kc: usize,
    },

    /// VNNI 4元素交织: 4个连续元素按列交织存储
    /// 用于 AVX-512 VNNI / ARM dotprod 指令
    /// 原始: W[row][col] (RowMajor)
    /// VNNI: W_vnni[row/4][col][lane] = W[row][col] where lane = row % 4
    VnniPack {
        /// 交织宽度 (4 for VNNI, 2 for VNNI-BF16)
        interleave: usize,
    },

    /// Tile layout: 固定大小 tile 存储 (AMX 16×16, GPU tensor core 16×16)
    /// 原始: W[m][n]
    /// Tile: tile[m/tile_m][n/tile_n][r][c] = W[m*tile_m+r][n*tile_n+c]
    TilePack {
        tile_rows: usize,
        tile_cols: usize,
    },
}

impl PackMap {
    /// 从原始 (row, col) 计算虚拟 pack 后的字节偏移
    /// 编译时可内联为 VmInstr stride 计算
    pub fn map_offset(&self, row: usize, col: usize, orig_ldc: usize, elem_bytes: usize) -> usize {
        match self {
            PackMap::Identity => (row * orig_ldc + col) * elem_bytes,

            PackMap::PanelPack { nr, kc } => {
                let panel_j = col / nr;
                let intra_r = row % nr;
                let intra_c = col % nr;
                let panel_id = row / nr; // simplified: panel row
                // BLIS B-pack: panel[col_block][kc_block][r][c]
                // For single kc block: offset = panel_j * nr * kc + intra_r * kc + intra_c
                (panel_j * nr * kc + intra_r * kc + intra_c) * elem_bytes
            }

            PackMap::VnniPack { interleave } => {
                let group = row / interleave;
                let lane = row % interleave;
                (group * orig_ldc * interleave + col * interleave + lane) * elem_bytes
            }

            PackMap::TilePack { tile_rows, tile_cols } => {
                let tile_row = row / tile_rows;
                let tile_col = col / tile_cols;
                let intra_r = row % tile_rows;
                let intra_c = col % tile_cols;
                (tile_row * tile_cols + tile_col * tile_rows * tile_cols
                    + intra_r * tile_cols + intra_c) * elem_bytes
            }
        }
    }

    /// pack 后的总字节数
    pub fn packed_bytes(&self, rows: usize, cols: usize, elem_bytes: usize) -> usize {
        match self {
            PackMap::Identity => rows * cols * elem_bytes,
            PackMap::PanelPack { nr, kc } => {
                let num_panels_j = (cols + nr - 1) / nr;
                let num_panels_i = (rows + nr - 1) / nr;
                num_panels_i * num_panels_j * nr * kc * elem_bytes
            }
            PackMap::VnniPack { interleave } => rows * cols * elem_bytes, // same size, different layout
            PackMap::TilePack { tile_rows, tile_cols } => {
                let aligned_rows = ((rows + tile_rows - 1) / tile_rows) * tile_rows;
                let aligned_cols = ((cols + tile_cols - 1) / tile_cols) * tile_cols;
                aligned_rows * aligned_cols * elem_bytes
            }
        }
    }

    /// 是否需要物理 pack buffer (Identity 不需要)
    pub fn requires_physical_pack(&self) -> bool {
        !matches!(self, PackMap::Identity)
    }

    /// BLIS K 循环中 B-matrix 的行步进（字节偏移）。
    /// RowMajor: k_ctr * n * elem_bytes (每行 N 个元素)
    /// PanelPack: k_ctr * nr * elem_bytes (panel 内每行 nr 个元素)
    /// VnniPack: k_ctr * n * interleave * elem_bytes
    /// TilePack: k_ctr * tile_cols * elem_bytes
    pub fn blis_k_stride_bytes(&self, orig_n: usize, elem_bytes: usize) -> usize {
        match self {
            PackMap::Identity => orig_n * elem_bytes,
            PackMap::PanelPack { nr, kc: _ } => nr * elem_bytes,
            PackMap::VnniPack { interleave } => orig_n * interleave * elem_bytes,
            PackMap::TilePack { tile_rows: _, tile_cols } => tile_cols * elem_bytes,
        }
    }
}

/// 从 LayoutConstraint 推导 PackMap
pub fn pack_map_from_layout(
    layout: &crate::compiler::accel_registry::LayoutConstraint,
) -> PackMap {
    match layout {
        crate::compiler::accel_registry::LayoutConstraint::PanelPacked { mr: _, nr } => {
            // 默认 kc 从 DeviceProfile 获取，这里用 0 表示 "运行时确定"
            PackMap::PanelPack { nr: *nr, kc: 0 }
        }
        crate::compiler::accel_registry::LayoutConstraint::VnniPacked4 => {
            PackMap::VnniPack { interleave: 4 }
        }
        crate::compiler::accel_registry::LayoutConstraint::AmxTileBF16 { rows, cols } => {
            PackMap::TilePack { tile_rows: *rows, tile_cols: *cols }
        }
        _ => PackMap::Identity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_no_pack() {
        let pm = PackMap::Identity;
        assert!(!pm.requires_physical_pack());
        assert_eq!(pm.map_offset(2, 3, 8, 4), (2 * 8 + 3) * 4);
    }

    #[test]
    fn test_panel_pack_offset() {
        let pm = PackMap::PanelPack { nr: 4, kc: 16 };
        assert!(pm.requires_physical_pack());
        // row=0, col=0: panel_j=0, intra_r=0, intra_c=0 → offset=0
        assert_eq!(pm.map_offset(0, 0, 64, 4), 0);
        // row=1, col=0: panel_j=0, intra_r=1, intra_c=0 → 1*16*4 = 64
        assert_eq!(pm.map_offset(1, 0, 64, 4), 64);
        // row=0, col=1: panel_j=0, intra_r=0, intra_c=1 → 1*4 = 4
        assert_eq!(pm.map_offset(0, 1, 64, 4), 4);
    }

    #[test]
    fn test_vnni_pack() {
        let pm = PackMap::VnniPack { interleave: 4 };
        assert!(pm.requires_physical_pack());
        // Same total size
        assert_eq!(pm.packed_bytes(8, 8, 4), 8 * 8 * 4);
    }

    #[test]
    fn test_tile_pack() {
        let pm = PackMap::TilePack { tile_rows: 16, tile_cols: 16 };
        assert!(pm.requires_physical_pack());
        // Aligned to tile boundary
        assert_eq!(pm.packed_bytes(17, 17, 2), 32 * 32 * 2);
    }

    #[test]
    fn test_pack_map_from_layout() {
        use crate::compiler::accel_registry::LayoutConstraint;

        let pm = pack_map_from_layout(&LayoutConstraint::PanelPacked { mr: 14, nr: 32 });
        assert!(matches!(pm, PackMap::PanelPack { nr: 32, .. }));

        let pm = pack_map_from_layout(&LayoutConstraint::RowMajor { align_bytes: 64 });
        assert!(matches!(pm, PackMap::Identity));

        let pm = pack_map_from_layout(&LayoutConstraint::VnniPacked4);
        assert!(matches!(pm, PackMap::VnniPack { interleave: 4 }));
    }

    // ── Test 6: Identity packed_bytes ──

    #[test]
    fn identity_packed_bytes() {
        let pm = PackMap::Identity;
        assert_eq!(pm.packed_bytes(8, 16, 4), 8 * 16 * 4);
        assert_eq!(pm.packed_bytes(1, 1, 2), 2);
    }

    // ── Test 7: Identity blis_k_stride_bytes ──

    #[test]
    fn identity_blis_k_stride() {
        let pm = PackMap::Identity;
        assert_eq!(pm.blis_k_stride_bytes(64, 4), 64 * 4);
    }

    // ── Test 8: PanelPack blis_k_stride_bytes ──

    #[test]
    fn panel_pack_blis_k_stride() {
        let pm = PackMap::PanelPack { nr: 6, kc: 32 };
        assert_eq!(pm.blis_k_stride_bytes(64, 4), 6 * 4);
    }

    // ── Test 9: VnniPack blis_k_stride_bytes ──

    #[test]
    fn vnni_pack_blis_k_stride() {
        let pm = PackMap::VnniPack { interleave: 4 };
        assert_eq!(pm.blis_k_stride_bytes(32, 2), 32 * 4 * 2);
    }

    // ── Test 10: TilePack blis_k_stride_bytes ──

    #[test]
    fn tile_pack_blis_k_stride() {
        let pm = PackMap::TilePack { tile_rows: 16, tile_cols: 16 };
        assert_eq!(pm.blis_k_stride_bytes(64, 4), 16 * 4);
    }

    // ── Test 11: PackMap Debug format ──

    #[test]
    fn pack_map_debug_format() {
        assert!(format!("{:?}", PackMap::Identity).contains("Identity"));
        assert!(format!("{:?}", PackMap::VnniPack { interleave: 4 }).contains("VnniPack"));
        assert!(format!("{:?}", PackMap::TilePack { tile_rows: 16, tile_cols: 16 }).contains("TilePack"));
    }

    // ── Test 12: PackMap Clone and PartialEq ──

    #[test]
    fn pack_map_clone_and_equality() {
        let a = PackMap::PanelPack { nr: 4, kc: 16 };
        let b = a.clone();
        assert_eq!(a, b);

        let c = PackMap::PanelPack { nr: 4, kc: 32 };
        assert_ne!(a, c);
    }

    // ── Test 13: VnniPack map_offset ──

    #[test]
    fn vnni_pack_map_offset() {
        let pm = PackMap::VnniPack { interleave: 4 };
        let ldc = 8;
        // row=0, col=0: group=0, lane=0 → offset = 0*8*4 + 0*4 + 0 = 0
        assert_eq!(pm.map_offset(0, 0, ldc, 4), 0);
        // row=4, col=0: group=1, lane=0 → offset = 1*8*4*4 + 0*4 + 0 = 128
        assert_eq!(pm.map_offset(4, 0, ldc, 4), 1 * ldc * 4 * 4);
        // row=1, col=2: group=0, lane=1 → offset = 0*8*4*4 + 2*4 + 1 = 9 → *4 = 36
        assert_eq!(pm.map_offset(1, 2, ldc, 4), (0 * ldc * 4 + 2 * 4 + 1) * 4);
    }

    // ── Test 14: pack_map_from_layout AmxTileBF16 ──

    #[test]
    fn pack_map_from_layout_amx_tile() {
        use crate::compiler::accel_registry::LayoutConstraint;

        let pm = pack_map_from_layout(&LayoutConstraint::AmxTileBF16 { rows: 16, cols: 16 });
        assert!(matches!(pm, PackMap::TilePack { tile_rows: 16, tile_cols: 16 }));
    }

    // ── Test 15: TilePack map_offset ──

    #[test]
    fn tile_pack_map_offset() {
        let pm = PackMap::TilePack { tile_rows: 4, tile_cols: 4 };
        // row=0, col=0 → tile_row=0, tile_col=0, intra_r=0, intra_c=0
        assert_eq!(pm.map_offset(0, 0, 8, 4), 0);
        // row=4, col=0 → tile_row=1, tile_col=0, intra_r=0, intra_c=0
        // offset = (1*4 + 0*4*4 + 0*4 + 0) * 4 = 16
        assert_eq!(pm.map_offset(4, 0, 8, 4), 16);
    }

    // ── Additional tests ──

    #[test]
    fn identity_map_offset_various_elem_bytes() {
        let pm = PackMap::Identity;
        // elem_bytes = 1 (u8/int8)
        assert_eq!(pm.map_offset(3, 5, 10, 1), 3 * 10 + 5);
        // elem_bytes = 2 (f16/bf16)
        assert_eq!(pm.map_offset(3, 5, 10, 2), (3 * 10 + 5) * 2);
        // elem_bytes = 8 (f64)
        assert_eq!(pm.map_offset(2, 3, 8, 8), (2 * 8 + 3) * 8);
    }

    #[test]
    fn identity_packed_bytes_various_elem_bytes() {
        let pm = PackMap::Identity;
        assert_eq!(pm.packed_bytes(4, 8, 1), 32);
        assert_eq!(pm.packed_bytes(4, 8, 2), 64);
        assert_eq!(pm.packed_bytes(4, 8, 8), 256);
    }

    #[test]
    fn panel_pack_packed_bytes_rounds_up() {
        let pm = PackMap::PanelPack { nr: 4, kc: 16 };
        // cols=9, nr=4 -> num_panels_j = (9+3)/4 = 3
        // rows=5, nr=4 -> num_panels_i = (5+3)/4 = 2
        // total = 2 * 3 * 4 * 16 = 384 elements * 4 bytes = 1536
        assert_eq!(pm.packed_bytes(5, 9, 4), 2 * 3 * 4 * 16 * 4);
    }

    #[test]
    fn vnni_pack_map_offset_second_row_in_group() {
        let pm = PackMap::VnniPack { interleave: 4 };
        let ldc = 8;
        // row=2, col=3: group=0, lane=2 -> offset = (0*8*4 + 3*4 + 2) * elem_bytes
        let expected = (0 * ldc * 4 + 3 * 4 + 2) * 4;
        assert_eq!(pm.map_offset(2, 3, ldc, 4), expected);
    }

    #[test]
    fn tile_pack_map_offset_within_tile() {
        let pm = PackMap::TilePack { tile_rows: 8, tile_cols: 8 };
        // row=3, col=5: tile_row=0, tile_col=0, intra_r=3, intra_c=5
        // offset = (0*8 + 0*8*8 + 3*8 + 5) * 4 = 29*4 = 116
        assert_eq!(pm.map_offset(3, 5, 16, 4), (3 * 8 + 5) * 4);
    }

    #[test]
    fn tile_pack_packed_bytes_exact_multiple() {
        let pm = PackMap::TilePack { tile_rows: 16, tile_cols: 16 };
        // 32x32 is exact multiple of 16x16 -> no padding
        assert_eq!(pm.packed_bytes(32, 32, 4), 32 * 32 * 4);
    }

    #[test]
    fn pack_map_from_layout_col_major_is_identity() {
        use crate::compiler::accel_registry::LayoutConstraint;
        // ColMajor layout should map to Identity
        let pm = pack_map_from_layout(&LayoutConstraint::ColMajor { align_bytes: 64 });
        assert_eq!(pm, PackMap::Identity);
    }

    #[test]
    fn panel_pack_map_offset_cross_panel_boundary() {
        let pm = PackMap::PanelPack { nr: 4, kc: 16 };
        // col=4 is the start of the second panel (panel_j=1)
        // row=0, col=4: panel_j=1, intra_r=0, intra_c=0 -> offset = 1*4*16*4 = 256
        assert_eq!(pm.map_offset(0, 4, 64, 4), 1 * 4 * 16 * 4);
        // row=0, col=7: panel_j=1, intra_r=0, intra_c=3 -> offset = (1*4*16 + 0*16 + 3)*4 = (64+3)*4 = 268
        assert_eq!(pm.map_offset(0, 7, 64, 4), (4 * 16 + 3) * 4);
    }

    #[test]
    fn pack_map_debug_comprehensive() {
        // Verify Debug output contains variant names for all variants
        let identity_debug = format!("{:?}", PackMap::Identity);
        assert!(identity_debug.contains("Identity"));

        let panel_debug = format!("{:?}", PackMap::PanelPack { nr: 6, kc: 32 });
        assert!(panel_debug.contains("PanelPack"));
        assert!(panel_debug.contains("nr"));
        assert!(panel_debug.contains("kc"));
    }
}
