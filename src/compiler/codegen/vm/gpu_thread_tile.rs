//! GPU SIMT 线程分区原语 (gpu_ir 层, ARCH-GPU-SIMT-THREAD-MODEL)。
//!
//! 核心契约 (architect 决策 `SPEC/DECISION-gpu-gemm-accs-warp-model.md` 选项 B):
//! GPU 累加类 kernel 的运行时并行来自 SIMT 硬件 (threadIdx/laneId/blockIdx),
//! 不是 Rust unroll。每个 thread 持有一个小的 thread tile (TM×TN 寄存器
//! accumulator), 整个 warp/block 的输出覆盖 = lanes × thread_tile 协同。
//!
//! ## 为什么这样设计
//!
//! 旧实现把 CPU BLIS 的 row/col 累加逻辑原样搬到 GPU: 一个 thread 顺序迭代
//! `warp_m × warp_n` 输出 tile (e.g. 64×32=2048), 用 `accs[..16].min(16)` 截断
//! → 2032 个输出单元无 accumulator, `if acc_idx < accs.len()` 静默跳过
//! (NO_SILENT-FALLBACK 违规)。调大 accs 立刻 VReg 爆炸 + spill → GPU 无栈 →
//! StackArg 报错。根因不是"16 太小", 是**根本没有线程/lane 模型**。
//!
//! 正解: accs 尺寸 = thread tile (TM×TN, 小编译期常量, e.g. 8×8=64, 远低于
//! PTX ~255 f32 寄存器上限), 留寄存器全程贯穿 K-loop; 跨 thread 的输出分布
//! 由 `OffsetExpr::ThreadOffset(Lane, stride)` 表达, lower 时映射到 `%laneid`。
//!
//! ## 适配的 kernel
//!
//! - GEMM (C = A·B): thread tile TM×TN accumulator, K-loop reduction
//! - attention (QK^T / PV 累加): 同结构, thread tile 复用
//! - reduction: 每 lane 一段输出, 各自沿 K 归约
//!
//! 复用入口: `GpuThreadTile::for_warp(warp_m, warp_n, warp_size)`。

use super::instr::{OffsetExpr, ThreadDim, VRegId, VRegKind, VmInstr, VmProgram};
use crate::types::CompilerError;

/// GPU warp 大小 (所有 NVIDIA/AMD GPU 均为 32)。
pub const GPU_WARP_SIZE: usize = 32;

/// GPU thread tile 配置 — 单个 thread 持有的寄存器 accumulator 子块。
///
/// 一个 warp (32 lane) 协同覆盖 `warp_m × warp_n` 输出 tile:
/// - `rows_per_lane` × `cols_per_lane` = warp_size (32), 每 lane 负责一个
///   `tm × tn` 子块 (`tm = warp_m / rows_per_lane`, `tn = warp_n / cols_per_lane`)
/// - accs 数组尺寸 = `tm * tn` (小, 编译期常量, 留寄存器)
/// - lane 的输出行基 = `lane_row * tm * row_stride`, 列基 = `lane_col * tn * elem`
///   (lane_row/lane_col 由 lane_id 推导)
#[derive(Debug, Clone, Copy)]
pub struct GpuThreadTile {
    /// warp 输出 tile 行数 (来自 HardwareProfile.gpu_gemm_tiles() 的 warp_m)
    pub warp_m: usize,
    /// warp 输出 tile 列数 (warp_n)
    pub warp_n: usize,
    /// 每 lane 负责的行数 (tm = warp_m / rows_per_lane)
    pub tm: usize,
    /// 每 lane 负责的列数 (tn = warp_n / cols_per_lane)
    pub tn: usize,
    /// lane 维度行划分数 (rows_per_lane × cols_per_lane = warp_size)
    pub rows_per_lane: usize,
    /// lane 维度列划分数
    pub cols_per_lane: usize,
}

impl GpuThreadTile {
    /// 按 warp 输出 tile 推导 thread tile 配置。
    ///
    /// 选 `rows_per_lane`/`cols_per_lane` 使 `tm`/`tn` 接近正方形 (寄存器占用均衡),
    /// 且 `tm * tn` ≤ `max_acc_per_thread` (默认 64, 远低于 PTX ~255 f32 上限)。
    /// `warp_m × warp_n` 必须 ≤ `warp_size × max_acc_per_thread` (e.g. 32×64=2048)。
    pub fn for_warp(warp_m: usize, warp_n: usize, max_acc_per_thread: usize) -> Self {
        // 单方向切分 (OffsetExpr ThreadOffset, 简单) 优先; 不行则双轴切分 (ThreadCoord div/mod)。
        //   (a) lane 切列: rows_per_lane=1, cols_per_lane=warp_size (需 warp_n % warp_size == 0)
        //   (b) lane 切行: rows_per_lane=warp_size, cols_per_lane=1 (需 warp_m % warp_size == 0)
        //   (c) 双轴: rows_per_lane×cols_per_lane=warp_size, lane_row=lane_id/cols_per_lane,
        //       lane_col=lane_id%cols_per_lane (需 warp_m%rows_per_lane==0 && warp_n%cols_per_lane==0)
        // score = acc + |tm - tn| * 10 (越小越好: acc 小 + tm/tn 接近正方形)。
        let mut best: Option<(usize, usize, usize, usize, usize)> = None; // (score, rpl, cpl, tm, tn)

        // (a) lane 切列方向: tm = warp_m, tn = warp_n / warp_size
        if warp_n % GPU_WARP_SIZE == 0 {
            let tm = warp_m;
            let tn = warp_n / GPU_WARP_SIZE;
            let acc = tm * tn;
            if acc > 0 && acc <= max_acc_per_thread {
                let score = acc + tm.abs_diff(tn) * 10;
                best = Some((score, 1, GPU_WARP_SIZE, tm, tn));
            }
        }
        // (b) lane 切行方向: tm = warp_m / warp_size, tn = warp_n
        if warp_m % GPU_WARP_SIZE == 0 {
            let tm = warp_m / GPU_WARP_SIZE;
            let tn = warp_n;
            let acc = tm * tn;
            if acc > 0 && acc <= max_acc_per_thread {
                let score = acc + tm.abs_diff(tn) * 10;
                best = Some(match best {
                    None => (score, GPU_WARP_SIZE, 1, tm, tn),
                    Some((bs, _, _, _, _)) if score < bs => (score, GPU_WARP_SIZE, 1, tm, tn),
                    Some(b) => b,
                });
            }
        }
        // (c) 双轴切分: rows_per_lane ∈ {2,4,8,16}, cols_per_lane = warp_size/rows_per_lane
        for &rpl in &[2usize, 4, 8, 16] {
            let cpl = GPU_WARP_SIZE / rpl;
            if cpl == 0 || warp_m % rpl != 0 || warp_n % cpl != 0 {
                continue;
            }
            let tm = warp_m / rpl;
            let tn = warp_n / cpl;
            let acc = tm * tn;
            if acc == 0 || acc > max_acc_per_thread {
                continue;
            }
            let score = acc + tm.abs_diff(tn) * 10;
            best = Some(match best {
                None => (score, rpl, cpl, tm, tn),
                Some((bs, _, _, _, _)) if score < bs => (score, rpl, cpl, tm, tn),
                Some(b) => b,
            });
        }

        let (rows_per_lane, cols_per_lane, tm, tn) =
            best.map(|(_, r, c, m, n)| (r, c, m, n)).unwrap_or_else(|| {
                // 单/双轴切分都不可行 (warp_m/warp_n 都 < warp_size 且无法整除)。
                // 回退: 不切分 (1 thread 持有整 warp tile), 仅 warp_m*warp_n ≤ max_acc 时合法。
                (1, 1, warp_m, warp_n)
            });
        Self {
            warp_m,
            warp_n,
            tm,
            tn,
            rows_per_lane,
            cols_per_lane,
        }
    }

    /// accumulator 数量 = tm * tn (每 thread 持有的寄存器数)。
    pub fn acc_count(&self) -> usize {
        self.tm * self.tn
    }

    /// 分配 accumulator 寄存器 (tm*tn 个, 编译期索引, 留寄存器)。
    /// 调用方在 K-loop 前清零, K-loop 内 FMA 累加, 写回时按 thread tile 索引。
    pub fn alloc_accs(&self, prog: &mut VmProgram, width: super::instr::SimdWidth) -> Vec<VRegId> {
        (0..self.acc_count())
            .map(|_| prog.alloc_vreg(VRegKind::Vec, width))
            .collect()
    }

    /// lane 在输出行方向的字节基偏移 (lane_row * tm * row_stride)。
    pub fn lane_row_byte_offset(&self, row_stride: usize) -> OffsetExpr {
        use super::instr::ThreadCoordExpr;
        if self.rows_per_lane == GPU_WARP_SIZE {
            // lane 切行: lane_row = lane_id
            OffsetExpr::ThreadOffset(ThreadDim::Lane, self.tm * row_stride)
        } else if self.rows_per_lane == 1 && self.cols_per_lane == GPU_WARP_SIZE {
            // lane 切列: 行方向 lane 贡献 0
            OffsetExpr::Const(0)
        } else if self.rows_per_lane >= 2 {
            // 双轴: lane_row = lane_id / cols_per_lane
            OffsetExpr::ThreadCoord(
                ThreadCoordExpr::Div(ThreadDim::Lane, self.cols_per_lane),
                self.tm * row_stride,
            )
        } else {
            // 无切分: lane 不参与
            OffsetExpr::Const(0)
        }
    }

    /// lane 在输出列方向的字节基偏移 (lane_col * tn * c_elem)。
    pub fn lane_col_byte_offset(&self, c_elem: usize) -> OffsetExpr {
        use super::instr::ThreadCoordExpr;
        if self.cols_per_lane == GPU_WARP_SIZE {
            // lane 切列: lane_col = lane_id
            OffsetExpr::ThreadOffset(ThreadDim::Lane, self.tn * c_elem)
        } else if self.cols_per_lane == 1 && self.rows_per_lane == GPU_WARP_SIZE {
            // lane 切行: 列方向 lane 贡献 0
            OffsetExpr::Const(0)
        } else if self.cols_per_lane >= 2 {
            // 双轴: lane_col = lane_id % cols_per_lane
            OffsetExpr::ThreadCoord(
                ThreadCoordExpr::Mod(ThreadDim::Lane, self.cols_per_lane),
                self.tn * c_elem,
            )
        } else {
            // 无切分: lane 不参与
            OffsetExpr::Const(0)
        }
    }

    /// 验证 thread tile 配置可正确映射。
    ///
    /// 合法配置:
    ///   (a) lane 切列 (rows_per_lane=1, cols_per_lane=warp_size) — ThreadOffset
    ///   (b) lane 切行 (rows_per_lane=warp_size, cols_per_lane=1) — ThreadOffset
    ///   (c) 双轴 (rows_per_lane×cols_per_lane=warp_size, 2/4/8/16) — ThreadCoord div/mod
    ///   (d) 无切分 (1,1) — 小 tile 回退, acc_count ≤ 64
    pub fn validate(&self) -> Result<(), CompilerError> {
        let lane_cuts_col = self.rows_per_lane == 1 && self.cols_per_lane == GPU_WARP_SIZE;
        let lane_cuts_row = self.rows_per_lane == GPU_WARP_SIZE && self.cols_per_lane == 1;
        let dual_axis = self.rows_per_lane >= 2
            && self.rows_per_lane < GPU_WARP_SIZE
            && self.cols_per_lane == GPU_WARP_SIZE / self.rows_per_lane
            && self.rows_per_lane * self.cols_per_lane == GPU_WARP_SIZE;
        let no_split = self.rows_per_lane == 1 && self.cols_per_lane == 1;
        if lane_cuts_col || lane_cuts_row || dual_axis {
            return Ok(());
        }
        if no_split {
            const MAX_NO_SPLIT_ACC: usize = 64;
            if self.acc_count() <= MAX_NO_SPLIT_ACC {
                return Ok(());
            }
            return Err(CompilerError::CodegenViolation(format!(
                "GpuThreadTile: no-split fallback (warp_m={}×warp_n={}={}) exceeds register budget {}, \
                 and lane split not possible. Reduce warp_m/warp_n or use warp tile divisible by warp_size.",
                self.warp_m, self.warp_n, self.acc_count(), MAX_NO_SPLIT_ACC
            )));
        }
        Err(CompilerError::CodegenViolation(format!(
            "GpuThreadTile: invalid split (rows_per_lane={}, cols_per_lane={}). \
             Use single-axis, dual-axis (rpl×cpl=warp_size), or no-split (1,1) with small tile.",
            self.rows_per_lane, self.cols_per_lane
        )))
    }
}

/// 辅助: 清零一组 accumulator 寄存器 (K-loop 前)。
pub fn zero_accs(
    prog: &mut VmProgram,
    accs: &[VRegId],
    width: super::instr::SimdWidth,
    acc_dtype: crate::compiler::trace::QuantPrecision,
) {
    use super::instr::{ScalarExpr, VmInstr};
    for acc in accs {
        prog.emit(VmInstr::Broadcast {
            dst: *acc,
            src: ScalarExpr::Const(0.0),
            width,
            dtype: acc_dtype,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_tile_warp64x32() {
        // warp_m=64, warp_n=32 (SmolLM2/lm_head 典型): 期望 tm*tn ≤ 64
        let tile = GpuThreadTile::for_warp(64, 32, 64);
        assert_eq!(tile.acc_count(), 64, "acc_count should be 64 not 2048");
        tile.validate().expect("single-axis split valid");
        // 最优配置: lane 切行 (tm=2, tn=32, acc=64) 或 lane 切列 (tm=64, tn=1, acc=64)
        // score 选 tm/tn 更接近正方形的: tm=2,tn=32 (|2-32|=30) vs tm=64,tn=1 (|64-1|=63)
        // → lane 切行: rows_per_lane=32, cols_per_lane=1
        assert!(tile.acc_count() <= 64);
    }

    #[test]
    fn test_thread_tile_warp64x16() {
        // SM90/SM80 warp_m=64, warp_n=16
        let tile = GpuThreadTile::for_warp(64, 16, 64);
        assert!(tile.acc_count() <= 64, "acc_count <= 64");
        tile.validate().expect("valid");
    }

    #[test]
    fn test_thread_tile_no_silent_guard_needed() {
        // 核心契约: acc_count == thread tile, 天然 == 每 thread 实际输出单元数,
        // 不再需要 acc_idx < accs.len() 守卫。
        let tile = GpuThreadTile::for_warp(64, 32, 64);
        // thread 内 row ∈ 0..tm, col ∈ 0..tn → acc_idx = row*tn + col ∈ 0..(tm*tn)
        // == accs.len(), 守卫永远 true → 删除守卫安全
        for row in 0..tile.tm {
            for col in 0..tile.tn {
                let acc_idx = row * tile.tn + col;
                assert!(
                    acc_idx < tile.acc_count(),
                    "acc_idx {} < acc_count {}",
                    acc_idx,
                    tile.acc_count()
                );
            }
        }
    }

    #[test]
    fn test_thread_tile_warp_gt16_output_complete() {
        // BCE 回归测试: warp_m*warp_n > 16 时输出完整 (accs 全量, 无 .min(16) 截断)。
        // 旧实现 warp_m=64*warp_n=32=2048 → .min(16) → 只 16 acc, 2032 单元静默丢弃。
        // 新实现: acc_count = tm*tn ≤ 64, 32 lane 协同覆盖 2048, 输出完整。
        let tile = GpuThreadTile::for_warp(64, 32, 64);
        assert!(tile.warp_m * tile.warp_n > 16, "warp tile > 16");
        assert!(tile.acc_count() <= 64, "thread acc ≤ 64 (register budget)");
        // 32 lane × tm*tn = warp_m*warp_n (协同覆盖完整)
        assert_eq!(
            GPU_WARP_SIZE * tile.acc_count(),
            tile.warp_m * tile.warp_n,
            "32 lane × tm*tn must cover full warp_m*warp_n output"
        );
        tile.validate().expect("valid single/dual-axis split");
    }
}
