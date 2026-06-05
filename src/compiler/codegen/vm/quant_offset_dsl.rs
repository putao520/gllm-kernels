//! QuantOffsetDsl — 量化感知偏移计算 DSL (REQ-LC-004/007/008/009)
//!
//! 将量化数据访问的偏移计算建模为表达式 DSL，分离输入 block_bytes 和输出 compute_dtype elem_bytes。
//!
//! 核心约束 (REQ-LC-008):
//! - 输入偏移使用量化格式的 block_bytes (Q4_0=18)
//! - 输出偏移使用 compute_dtype 的 elem_bytes (通常 F32=4)
//! - 二者独立推导，禁止混用
//!
//! SPEC 25 §3.1: 所有偏移参数从 QuantFormatDescriptor 自动推导，消除手写公式。
//!
//! 使用方式:
//! ```ignore
//! let dsl = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
//! let scaled = QuantOffsetDsl::BinOp {
//!     op: BinOpKind::Mul,
//!     lhs: Box::new(dsl),
//!     rhs: Box::new(QuantOffsetDsl::Const(2)),
//! };
//! ```

/// 二元运算种类 — 用于 QuantOffsetDsl::BinOp 的偏移表达式组合。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Shl,
    Shr,
    And,
    Or,
}

/// 量化感知偏移计算 DSL (REQ-LC-004/007/008)
///
/// 将偏移计算抽象为表达式树，确保输入偏移和输出偏移独立推导：
/// - `BlockIndex` / `ScaleOffset` / `ZeroPointOffset` / `DataOffset` 使用 `block_bytes` 推导输入偏移
/// - `OutputBlockOffset` / `OutputRowStride` 使用 `compute_elem_bytes` 推导输出偏移
/// - `BinOp` 允许组合多个偏移表达式
///
/// REQ-LC-008: 输入 vs 输出偏移严格分离:
/// | 偏移类型   | 计算方式                                        | 来源                |
/// |-----------|------------------------------------------------|---------------------|
/// | 输入(量化) | block_idx * block_bytes                        | QuantFormatDescriptor|
/// | 输入(scale)| block_base + scale_offset_bytes                | QuantFormatDescriptor|
/// | 输入(data) | block_base + data_offset + sub_idx * advance   | QuantFormatDescriptor|
/// | 输出       | block_idx * block_size * compute_elem_bytes    | compute_dtype        |
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuantOffsetDsl {
    // ── 输入偏移 (使用 block_bytes / 量化格式参数) ──────────────────────

    /// 常量偏移值
    Const(i64),

    /// 第 n 个 block 的起始偏移: n * block_bytes
    BlockIndex {
        block_bytes: usize,
    },

    /// scale 数组中的偏移: n * block_bytes + scale_offset * scale_elem_bytes
    ScaleOffset {
        block_bytes: usize,
        scale_elem_bytes: usize,
    },

    /// zero_point 数组中的偏移: n * block_bytes + zp_offset * zp_elem_bytes
    ZeroPointOffset {
        block_bytes: usize,
        zp_elem_bytes: usize,
    },

    /// data 在 block 内的偏移: sub_block_idx * data_byte_advance
    /// 用于 QuantGather/QuantGemm 内部 data_ptr 的初始偏移和步进计算。
    DataOffset {
        /// QuantFormatDescriptor.data_layout 推导的每 sub-block 字节步进
        data_byte_advance: usize,
        /// sub_block 索引 (通常为 0 表示初始偏移)
        sub_block_idx: usize,
    },

    // ── 输出偏移 (使用 compute_dtype.elem_bytes) ─────────────────────

    /// 输出缓冲区 block 级偏移: block_idx * block_size * compute_elem_bytes
    /// REQ-LC-008: 输出偏移使用 compute_dtype.elem_bytes(), 不是 block_bytes
    OutputBlockOffset {
        block_size: usize,
        compute_elem_bytes: usize,
    },

    /// 输出行步进: hidden_dim * compute_elem_bytes
    OutputRowStride {
        hidden_dim: usize,
        compute_elem_bytes: usize,
    },

    /// sub-block 输出步进: lanes * compute_elem_bytes
    SubBlockOutputStep {
        lanes: usize,
        compute_elem_bytes: usize,
    },

    // ── 组合运算 ───────────────────────────────────────────────────────

    /// 二元运算组合
    BinOp {
        op: BinOpKind,
        lhs: Box<QuantOffsetDsl>,
        rhs: Box<QuantOffsetDsl>,
    },
}

/// 编译期量化偏移计算上下文 (REQ-LC-005)
///
/// 包含编译时已知的所有索引变量，用于将 `QuantOffsetDsl` 求值为具体 i64 偏移值。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantOffsetContext {
    /// 当前 block 在 tensor 中的索引
    pub block_idx: usize,
    /// 当前 layer 在模型中的索引
    pub layer_idx: usize,
    /// 当前 head 在 multi-head attention 中的索引
    pub head_idx: usize,
    /// 当前 sub-block 索引 (用于 DataOffset 计算)
    pub sub_block_idx: usize,
}

impl QuantOffsetContext {
    /// 创建编译期量化偏移计算上下文。
    pub fn new(block_idx: usize, layer_idx: usize, head_idx: usize) -> Self {
        Self {
            block_idx,
            layer_idx,
            head_idx,
            sub_block_idx: 0,
        }
    }

    /// 创建带 sub_block_idx 的上下文。
    pub fn with_sub_block(block_idx: usize, sub_block_idx: usize) -> Self {
        Self {
            block_idx,
            layer_idx: 0,
            head_idx: 0,
            sub_block_idx,
        }
    }
}

/// 编译期求值 `QuantOffsetDsl` 为具体 i64 偏移值 (REQ-LC-005)
///
/// 使用 `QuantOffsetContext` 提供的编译时已知变量（block_idx、layer_idx、head_idx），
/// 递归求值 `QuantOffsetDsl` 表达式树。对于仅依赖 `block_idx` 的变体，委托给
/// `evaluate(block_idx)`；对于需要更丰富上下文的场景，`layer_idx` 和 `head_idx`
/// 为未来扩展保留。
pub fn eval_quant_offset(dsl: &QuantOffsetDsl, context: &QuantOffsetContext) -> i64 {
    dsl.evaluate_with_context(context)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// QuantOffsetDsl derive helpers from QuantFormatDescriptor (REQ-LC-007~009)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl QuantOffsetDsl {
    // ── 输入偏移 derive (REQ-LC-008: 使用 block_bytes) ──────────────────

    /// 从 QuantFormatDescriptor 推导量化 block 的字节步进 (REQ-LC-007).
    /// 输入偏移: desc.block_bytes.
    pub fn derive_block_stride(desc: &crate::quant_format::QuantFormatDescriptor) -> Self {
        QuantOffsetDsl::Const(desc.block_bytes as i64)
    }

    /// 从 QuantFormatDescriptor 推导每 sub-block 内 data 指针的字节步进 (REQ-LC-007).
    /// 输入偏移: 由 data_layout 决定.
    pub fn derive_data_byte_advance(
        desc: &crate::quant_format::QuantFormatDescriptor,
        lanes: usize,
    ) -> i64 {
        use crate::quant_format::DataLayout;
        match &desc.data_layout {
            DataLayout::PackedNibbles { .. } => (lanes / 2) as i64,
            DataLayout::Bytes { .. } => lanes as i64,
            DataLayout::NibbleWithHighBits { .. } => (lanes / 2) as i64,
            DataLayout::CodebookIndex { index_bits, .. } =>
                ((lanes * (*index_bits as usize) + 7) / 8) as i64,
        }
    }

    /// 行字节数 (量化格式) — 一行 hidden_dim 个元素占用多少字节.
    /// 输入偏移: (hidden_dim / block_size) * block_bytes.
    pub fn derive_row_stride_bytes(desc: &crate::quant_format::QuantFormatDescriptor, hidden_dim: usize) -> usize {
        (hidden_dim / desc.block_size) * desc.block_bytes
    }

    /// 从 QuantFormatDescriptor 推导 data 在 block 内的初始偏移 (REQ-LC-007).
    /// 输入偏移: sub_block_idx * data_byte_advance.
    pub fn derive_data_offset(
        desc: &crate::quant_format::QuantFormatDescriptor,
        sub_block_idx: usize,
        lanes: usize,
    ) -> Self {
        let advance = Self::derive_data_byte_advance(desc, lanes);
        QuantOffsetDsl::DataOffset {
            data_byte_advance: advance as usize,
            sub_block_idx,
        }
    }

    /// 从 QuantFormatDescriptor 推导 scale 在 block 内的偏移 (REQ-LC-007).
    pub fn derive_scale_offset(desc: &crate::quant_format::QuantFormatDescriptor) -> usize {
        use crate::quant_format::ScaleLayout;
        match &desc.scale_layout {
            ScaleLayout::BlockScalar { offset_bytes, .. } => *offset_bytes,
            ScaleLayout::BlockScalarWithMin { d_offset, .. } => *d_offset,
            _ => 0,
        }
    }

    // ── 输出偏移 derive (REQ-LC-008: 使用 compute_elem_bytes) ────────────

    /// 输出行步进字节数 — 输出偏移 (REQ-LC-008).
    /// 输出偏移: hidden_dim * compute_elem_bytes.
    pub fn derive_output_row_stride(hidden_dim: usize, compute_elem_bytes: usize) -> Self {
        QuantOffsetDsl::OutputRowStride {
            hidden_dim,
            compute_elem_bytes,
        }
    }

    /// 输出 block 级偏移: block_idx * block_size * compute_elem_bytes (REQ-LC-008).
    pub fn derive_output_block_offset(
        block_size: usize,
        compute_elem_bytes: usize,
    ) -> Self {
        QuantOffsetDsl::OutputBlockOffset {
            block_size,
            compute_elem_bytes,
        }
    }

    /// sub-block 输出步进: lanes * compute_elem_bytes (REQ-LC-008).
    pub fn derive_sub_block_output_step(lanes: usize, compute_elem_bytes: usize) -> Self {
        QuantOffsetDsl::SubBlockOutputStep {
            lanes,
            compute_elem_bytes,
        }
    }

    // ── 辅助方法 ───────────────────────────────────────────────────────

    /// 每个 block 包含的 sub-block 数量.
    pub fn sub_block_count(
        desc: &crate::quant_format::QuantFormatDescriptor,
        lanes: usize,
    ) -> usize {
        desc.block_size / lanes
    }

    // ── 求值 ────────────────────────────────────────────────────────────

    /// 计算偏移表达式的值（在已知 block_index 时求值）。
    ///
    /// 对于不需要运行时求值的场景（如编译时已知循环步进），
    /// 可以通过此方法计算常量折叠后的偏移值。
    pub fn evaluate(&self, block_index: usize) -> i64 {
        self.evaluate_with_context(&QuantOffsetContext::new(block_index, 0, 0))
    }

    /// 使用完整上下文求值 (REQ-LC-007).
    fn evaluate_with_context(&self, ctx: &QuantOffsetContext) -> i64 {
        match self {
            QuantOffsetDsl::Const(v) => *v,
            QuantOffsetDsl::BlockIndex { block_bytes } => {
                (ctx.block_idx * block_bytes) as i64
            }
            QuantOffsetDsl::ScaleOffset { block_bytes, scale_elem_bytes } => {
                (ctx.block_idx * block_bytes + scale_elem_bytes) as i64
            }
            QuantOffsetDsl::ZeroPointOffset { block_bytes, zp_elem_bytes } => {
                (ctx.block_idx * block_bytes + zp_elem_bytes) as i64
            }
            QuantOffsetDsl::DataOffset { data_byte_advance, sub_block_idx } => {
                (*sub_block_idx * data_byte_advance) as i64
            }
            // 输出偏移 (REQ-LC-008): 使用 compute_elem_bytes, 不是 block_bytes
            QuantOffsetDsl::OutputBlockOffset { block_size, compute_elem_bytes } => {
                (ctx.block_idx * block_size * compute_elem_bytes) as i64
            }
            QuantOffsetDsl::OutputRowStride { hidden_dim, compute_elem_bytes } => {
                (*hidden_dim * compute_elem_bytes) as i64
            }
            QuantOffsetDsl::SubBlockOutputStep { lanes, compute_elem_bytes } => {
                (*lanes * compute_elem_bytes) as i64
            }
            QuantOffsetDsl::BinOp { op, lhs, rhs } => {
                let lv = lhs.evaluate_with_context(ctx);
                let rv = rhs.evaluate_with_context(ctx);
                match op {
                    BinOpKind::Add => lv + rv,
                    BinOpKind::Sub => lv - rv,
                    BinOpKind::Mul => lv * rv,
                    BinOpKind::Div => {
                        if rv == 0 { 0 } else { lv / rv }
                    }
                    BinOpKind::Shl => lv << rv,
                    BinOpKind::Shr => lv >> rv,
                    BinOpKind::And => lv & rv,
                    BinOpKind::Or => lv | rv,
                }
            }
        }
    }

    /// 获取输入偏移的 block_bytes（如果是 BlockIndex / ScaleOffset / ZeroPointOffset 变体）。
    pub fn block_bytes(&self) -> Option<usize> {
        match self {
            QuantOffsetDsl::BlockIndex { block_bytes } => Some(*block_bytes),
            QuantOffsetDsl::ScaleOffset { block_bytes, .. } => Some(*block_bytes),
            QuantOffsetDsl::ZeroPointOffset { block_bytes, .. } => Some(*block_bytes),
            QuantOffsetDsl::BinOp { lhs, .. } => lhs.block_bytes(),
            _ => None,
        }
    }

    /// 判断此偏移是否为常量（不含 BlockIndex/ScaleOffset/ZeroPointOffset）。
    pub fn is_const(&self) -> bool {
        matches!(self, QuantOffsetDsl::Const(_))
    }

    /// 判断此偏移是否为输出偏移类型 (REQ-LC-008).
    pub fn is_output_offset(&self) -> bool {
        matches!(
            self,
            QuantOffsetDsl::OutputBlockOffset { .. }
                | QuantOffsetDsl::OutputRowStride { .. }
                | QuantOffsetDsl::SubBlockOutputStep { .. }
        )
    }

    /// 判断此偏移是否为输入偏移类型 (REQ-LC-008).
    pub fn is_input_offset(&self) -> bool {
        matches!(
            self,
            QuantOffsetDsl::BlockIndex { .. }
                | QuantOffsetDsl::ScaleOffset { .. }
                | QuantOffsetDsl::ZeroPointOffset { .. }
                | QuantOffsetDsl::DataOffset { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_offset() {
        let off = QuantOffsetDsl::Const(42);
        assert_eq!(off.evaluate(0), 42);
        assert_eq!(off.evaluate(5), 42);
        assert!(off.is_const());
    }

    #[test]
    fn test_block_index_offset() {
        let off = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
        assert_eq!(off.evaluate(0), 0);
        assert_eq!(off.evaluate(1), 18);
        assert_eq!(off.evaluate(3), 54);
        assert_eq!(off.block_bytes(), Some(18));
        assert!(!off.is_const());
        assert!(off.is_input_offset());
        assert!(!off.is_output_offset());
    }

    #[test]
    fn test_scale_offset() {
        let off = QuantOffsetDsl::ScaleOffset { block_bytes: 18, scale_elem_bytes: 2 };
        assert_eq!(off.evaluate(0), 2);
        assert_eq!(off.evaluate(1), 20);
        assert_eq!(off.evaluate(3), 56);
        assert_eq!(off.block_bytes(), Some(18));
    }

    #[test]
    fn test_zero_point_offset() {
        let off = QuantOffsetDsl::ZeroPointOffset { block_bytes: 18, zp_elem_bytes: 1 };
        assert_eq!(off.evaluate(0), 1);
        assert_eq!(off.evaluate(1), 19);
        assert_eq!(off.evaluate(3), 55);
        assert_eq!(off.block_bytes(), Some(18));
    }

    #[test]
    fn test_data_offset() {
        let off = QuantOffsetDsl::DataOffset { data_byte_advance: 4, sub_block_idx: 3 };
        assert_eq!(off.evaluate(0), 12); // 3 * 4
        assert_eq!(off.evaluate(5), 12); // block_idx irrelevant for DataOffset
        assert!(off.is_input_offset());
    }

    #[test]
    fn test_output_block_offset() {
        let off = QuantOffsetDsl::OutputBlockOffset { block_size: 32, compute_elem_bytes: 4 };
        assert_eq!(off.evaluate(0), 0);
        assert_eq!(off.evaluate(1), 128); // 1 * 32 * 4
        assert_eq!(off.evaluate(3), 384); // 3 * 32 * 4
        assert!(off.is_output_offset());
        assert!(!off.is_input_offset());
    }

    #[test]
    fn test_output_row_stride() {
        let off = QuantOffsetDsl::OutputRowStride { hidden_dim: 4096, compute_elem_bytes: 4 };
        assert_eq!(off.evaluate(0), 16384); // 4096 * 4
        assert!(off.is_output_offset());
    }

    #[test]
    fn test_sub_block_output_step() {
        let off = QuantOffsetDsl::SubBlockOutputStep { lanes: 8, compute_elem_bytes: 4 };
        assert_eq!(off.evaluate(0), 32); // 8 * 4
        assert!(off.is_output_offset());
    }

    #[test]
    fn test_binop_add() {
        let lhs = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
        let rhs = QuantOffsetDsl::Const(4);
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
        assert_eq!(off.evaluate(0), 4);
        assert_eq!(off.evaluate(1), 22);
        assert_eq!(off.evaluate(3), 58);
        assert_eq!(off.block_bytes(), Some(18));
    }

    #[test]
    fn test_binop_mul() {
        let lhs = QuantOffsetDsl::Const(3);
        let rhs = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
        assert_eq!(off.evaluate(0), 0);
        assert_eq!(off.evaluate(1), 54);
        assert_eq!(off.evaluate(2), 108);
    }

    #[test]
    fn test_nested_binop() {
        // (BlockIndex + Const(2)) * Const(4)
        let inner = QuantOffsetDsl::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(QuantOffsetDsl::BlockIndex { block_bytes: 18 }),
            rhs: Box::new(QuantOffsetDsl::Const(2)),
        };
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(inner),
            rhs: Box::new(QuantOffsetDsl::Const(4)),
        };
        assert_eq!(off.evaluate(0), 8);     // (0+2)*4
        assert_eq!(off.evaluate(1), 80);    // (18+2)*4
        assert_eq!(off.evaluate(3), 224);   // (54+2)*4
    }

    #[test]
    fn test_binop_with_div_by_zero() {
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Div,
            lhs: Box::new(QuantOffsetDsl::Const(100)),
            rhs: Box::new(QuantOffsetDsl::Const(0)),
        };
        assert_eq!(off.evaluate(0), 0); // div by zero -> 0
    }

    #[test]
    fn test_quant_offset_context_new() {
        let ctx = QuantOffsetContext::new(3, 1, 8);
        assert_eq!(ctx.block_idx, 3);
        assert_eq!(ctx.layer_idx, 1);
        assert_eq!(ctx.head_idx, 8);
    }

    #[test]
    fn test_eval_quant_offset_const() {
        let dsl = QuantOffsetDsl::Const(42);
        let ctx = QuantOffsetContext::new(5, 0, 0);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 42);
    }

    #[test]
    fn test_eval_quant_offset_block_index() {
        let dsl = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
        let ctx = QuantOffsetContext::new(2, 0, 0);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 36);
    }

    #[test]
    fn test_eval_quant_offset_scale_offset() {
        let dsl = QuantOffsetDsl::ScaleOffset { block_bytes: 18, scale_elem_bytes: 2 };
        let ctx = QuantOffsetContext::new(1, 0, 0);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 20);
    }

    #[test]
    fn test_eval_quant_offset_zero_point_offset() {
        let dsl = QuantOffsetDsl::ZeroPointOffset { block_bytes: 18, zp_elem_bytes: 1 };
        let ctx = QuantOffsetContext::new(3, 0, 0);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 55);
    }

    #[test]
    fn test_eval_quant_offset_binop() {
        let lhs = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
        let rhs = QuantOffsetDsl::Const(4);
        let dsl = QuantOffsetDsl::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
        let ctx = QuantOffsetContext::new(1, 0, 0);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 22);
    }

    #[test]
    fn test_eval_quant_offset_layer_head_context() {
        let ctx = QuantOffsetContext::new(2, 5, 12);
        assert_eq!(ctx.layer_idx, 5);
        assert_eq!(ctx.head_idx, 12);
        let dsl = QuantOffsetDsl::BlockIndex { block_bytes: 18 };
        assert_eq!(eval_quant_offset(&dsl, &ctx), 36);
    }

    #[test]
    fn test_derive_output_block_offset_evaluates_correctly() {
        let dsl = QuantOffsetDsl::derive_output_block_offset(32, 4);
        let ctx = QuantOffsetContext::new(5, 0, 0);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 640); // 5 * 32 * 4
    }

    #[test]
    fn test_derive_sub_block_output_step_evaluates_correctly() {
        let dsl = QuantOffsetDsl::derive_sub_block_output_step(8, 4);
        assert_eq!(dsl.evaluate(0), 32); // 8 * 4
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn test_binop_sub() {
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Sub,
            lhs: Box::new(QuantOffsetDsl::BlockIndex { block_bytes: 18 }),
            rhs: Box::new(QuantOffsetDsl::Const(4)),
        };
        assert_eq!(off.evaluate(0), -4);
        assert_eq!(off.evaluate(1), 14); // 18 - 4
        assert_eq!(off.evaluate(3), 50); // 54 - 4
    }

    #[test]
    fn test_binop_div() {
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Div,
            lhs: Box::new(QuantOffsetDsl::BlockIndex { block_bytes: 18 }),
            rhs: Box::new(QuantOffsetDsl::Const(3)),
        };
        assert_eq!(off.evaluate(0), 0);
        assert_eq!(off.evaluate(6), 36); // 6*18 = 108 / 3 = 36
    }

    #[test]
    fn test_binop_shl_shr() {
        let shl = QuantOffsetDsl::BinOp {
            op: BinOpKind::Shl,
            lhs: Box::new(QuantOffsetDsl::Const(1)),
            rhs: Box::new(QuantOffsetDsl::Const(4)),
        };
        assert_eq!(shl.evaluate(0), 16); // 1 << 4

        let shr = QuantOffsetDsl::BinOp {
            op: BinOpKind::Shr,
            lhs: Box::new(QuantOffsetDsl::Const(64)),
            rhs: Box::new(QuantOffsetDsl::Const(2)),
        };
        assert_eq!(shr.evaluate(0), 16); // 64 >> 2
    }

    #[test]
    fn test_binop_and_or() {
        let and_off = QuantOffsetDsl::BinOp {
            op: BinOpKind::And,
            lhs: Box::new(QuantOffsetDsl::Const(0xFF)),
            rhs: Box::new(QuantOffsetDsl::Const(0x0F)),
        };
        assert_eq!(and_off.evaluate(0), 0x0F);

        let or_off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Or,
            lhs: Box::new(QuantOffsetDsl::Const(0xF0)),
            rhs: Box::new(QuantOffsetDsl::Const(0x0F)),
        };
        assert_eq!(or_off.evaluate(0), 0xFF);
    }

    #[test]
    fn test_block_bytes_returns_none_for_non_block_variants() {
        assert!(QuantOffsetDsl::Const(5).block_bytes().is_none());
        assert!(QuantOffsetDsl::OutputBlockOffset { block_size: 32, compute_elem_bytes: 4 }.block_bytes().is_none());
        assert!(QuantOffsetDsl::OutputRowStride { hidden_dim: 512, compute_elem_bytes: 4 }.block_bytes().is_none());
        assert!(QuantOffsetDsl::SubBlockOutputStep { lanes: 8, compute_elem_bytes: 4 }.block_bytes().is_none());
        assert!(QuantOffsetDsl::DataOffset { data_byte_advance: 4, sub_block_idx: 0 }.block_bytes().is_none());
    }

    #[test]
    fn test_is_const_false_for_all_non_const() {
        assert!(!QuantOffsetDsl::BlockIndex { block_bytes: 18 }.is_const());
        assert!(!QuantOffsetDsl::ScaleOffset { block_bytes: 18, scale_elem_bytes: 2 }.is_const());
        assert!(!QuantOffsetDsl::ZeroPointOffset { block_bytes: 18, zp_elem_bytes: 1 }.is_const());
        assert!(!QuantOffsetDsl::OutputBlockOffset { block_size: 32, compute_elem_bytes: 4 }.is_const());
    }

    #[test]
    fn test_quant_offset_context_with_sub_block() {
        let ctx = QuantOffsetContext::with_sub_block(5, 3);
        assert_eq!(ctx.block_idx, 5);
        assert_eq!(ctx.layer_idx, 0);
        assert_eq!(ctx.head_idx, 0);
        assert_eq!(ctx.sub_block_idx, 3);
    }

    #[test]
    fn test_quant_offset_context_copy() {
        let ctx = QuantOffsetContext::new(1, 2, 3);
        let ctx2 = ctx; // Copy
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn test_quant_offset_context_hash_and_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QuantOffsetContext::new(1, 2, 3));
        set.insert(QuantOffsetContext::new(1, 2, 3));
        set.insert(QuantOffsetContext::new(4, 5, 6));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_derive_output_row_stride() {
        let dsl = QuantOffsetDsl::derive_output_row_stride(4096, 4);
        assert_eq!(dsl.evaluate(0), 16384); // 4096 * 4
        assert!(dsl.is_output_offset());
    }

    #[test]
    fn test_eval_quant_offset_data_offset_with_context() {
        let dsl = QuantOffsetDsl::DataOffset { data_byte_advance: 4, sub_block_idx: 2 };
        let ctx = QuantOffsetContext::with_sub_block(10, 2);
        assert_eq!(eval_quant_offset(&dsl, &ctx), 8); // 2 * 4
    }

    #[test]
    fn test_binop_left_delegates_block_bytes() {
        // BinOp delegates block_bytes() to lhs
        let off = QuantOffsetDsl::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(QuantOffsetDsl::BlockIndex { block_bytes: 18 }),
            rhs: Box::new(QuantOffsetDsl::Const(4)),
        };
        assert_eq!(off.block_bytes(), Some(18));

        // BinOp with Const lhs has no block_bytes
        let off2 = QuantOffsetDsl::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(QuantOffsetDsl::Const(4)),
            rhs: Box::new(QuantOffsetDsl::Const(4)),
        };
        assert_eq!(off2.block_bytes(), None);
    }

    #[test]
    fn test_const_negative_value() {
        let off = QuantOffsetDsl::Const(-42);
        assert_eq!(off.evaluate(0), -42);
        assert_eq!(off.evaluate(100), -42);
        assert!(off.is_const());
    }

    #[test]
    fn test_binop_kind_variants_equality() {
        assert_eq!(BinOpKind::Add, BinOpKind::Add);
        assert_ne!(BinOpKind::Add, BinOpKind::Sub);
        assert_ne!(BinOpKind::Mul, BinOpKind::Div);
        assert_ne!(BinOpKind::Shl, BinOpKind::Shr);
        assert_ne!(BinOpKind::And, BinOpKind::Or);
    }

    #[test]
    fn test_quant_offset_dsl_clone_preserves_equality() {
        let dsl = QuantOffsetDsl::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(QuantOffsetDsl::BlockIndex { block_bytes: 18 }),
            rhs: Box::new(QuantOffsetDsl::Const(2)),
        };
        let cloned = dsl.clone();
        assert_eq!(dsl, cloned);
        assert_eq!(cloned.evaluate(3), 108); // 3*18*2
    }
}
