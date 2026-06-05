//! DataKind 分族 trace 模板 (SPEC 24-QUANT-PIPELINE-JIT REQ-QPJ-003)
//!
//! 按 DataKind (Weight / Activation / KvCache / Rope) 分族生成 trace 模板。
//! 每个 DataKind 对应一类量化数据的加载/反量化/计算 trace。
//!
//! 模板参数化: quant_type, block_size, dtype → 生成 Vec<TraceOp> body。
//! auto_select 消费生成的 trace body 并展开为 VM 指令序列。

use crate::compiler::trace::{TraceOp, ValueId};
use crate::quant::QuantType;
use crate::quant_format::{QuantFormatDescriptor, ZeroLayout};

/// 数据分族分类 — 驱动 trace 模板选择和 auto_select 展开策略。
///
/// 每种 DataKind 对应一个语义角色，决定:
/// - 使用哪种 TraceOp 组合（QuantGemm / QuantGather / 纯加载 / 特化逻辑）
/// - block 解码参数（block_size, scale/zero layout）
/// - 内存访问模式（权重按行、激活按 tile、KV cache 按 slot）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataKind {
    /// 量化权重: GEMM/GEMV 中的 B 矩阵（模型权重）。
    /// Trace 模板: QuantGemm — block 循环加载 + 反量化 + FMA 累加。
    /// 典型来源: Linear 层的 weight tensor。
    Weight,

    /// 量化激活: GEMM 中的 A 矩阵或独立的逐元素操作输入。
    /// Trace 模板: 量化 block 加载 + 反量化 → F32 中间结果。
    /// 典型来源: 动态量化后的 hidden_states / intermediate 激活。
    Activation,

    /// KV cache: 注意力层缓存中的 K/V 张量。
    /// Trace 模板: slot 加载 + 按 page 解码（可能跨 page boundary）。
    /// 典型来源: paged KV cache 的 K/V buffer。
    KvCache,

    /// RoPE 位置编码: 旋转位置编码中的 cos/sin 查找表。
    /// Trace 模板: position index 加载 + cos/sin lookup + rotate。
    /// 典型来源: precomputed freqs_cis / RoPE lookup table。
    Rope,
}

/// 模板输出数据类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantOutputDtype {
    /// 反量化输出 F32。
    F32,
    /// 反量化输出 F16。
    F16,
    /// 反量化输出 BF16。
    BF16,
}

impl QuantOutputDtype {
    /// 每个 scalar 的字节数。
    pub fn elem_bytes(self) -> usize {
        match self {
            QuantOutputDtype::F32 => 4,
            QuantOutputDtype::F16 => 2,
            QuantOutputDtype::BF16 => 2,
        }
    }
}

/// 模板参数 — 驱动 trace body 生成的完整参数集。
#[derive(Debug, Clone)]
pub struct TraceTemplateParams {
    /// 目标数据分族。
    pub data_kind: DataKind,
    /// 量化格式类型。
    pub quant_type: QuantType,
    /// 量化 block 大小（每个 block 包含的元素数）。
    pub block_size: usize,
    /// 输出数据类型（通常为 F32，反量化后）。
    pub dtype: QuantOutputDtype,
    /// 可选: 量化格式描述符（当可用时提供精确的 scale/zero/data layout）。
    pub format_desc: Option<QuantFormatDescriptor>,
}

/// 从 QuantFormatDescriptor 的 scale_layout 提取 scale 字节偏移。
/// 返回 (offset_bytes, quant_type) 对。
fn scale_offset_from_desc(desc: &QuantFormatDescriptor) -> usize {
    match &desc.scale_layout {
        crate::quant_format::ScaleLayout::None => 0,
        crate::quant_format::ScaleLayout::BlockScalar { offset_bytes, .. } => *offset_bytes,
        crate::quant_format::ScaleLayout::BlockScalarWithMin { d_offset, .. } => *d_offset,
        crate::quant_format::ScaleLayout::Hierarchical { block_d_offset, .. } => *block_d_offset,
        crate::quant_format::ScaleLayout::Q6KScales { block_d_offset, .. } => *block_d_offset,
        crate::quant_format::ScaleLayout::ExternalArray { .. } => 0,
        crate::quant_format::ScaleLayout::SubBlockScalars { offset_bytes, .. } => *offset_bytes,
    }
}

/// 从 QuantFormatDescriptor 的 data_layout 提取 data 字节偏移。
fn data_offset_from_desc(desc: &QuantFormatDescriptor) -> usize {
    match &desc.data_layout {
        crate::quant_format::DataLayout::PackedNibbles { offset, .. } => *offset,
        crate::quant_format::DataLayout::NibbleWithHighBits { low_offset, .. } => *low_offset,
        crate::quant_format::DataLayout::Bytes { offset, .. } => *offset,
        crate::quant_format::DataLayout::CodebookIndex { offset, .. } => *offset,
    }
}

/// 从 QuantFormatDescriptor 的 zero_layout 提取 zero-point 字节偏移。
fn zero_offset_from_desc(desc: &QuantFormatDescriptor) -> usize {
    match &desc.zero_layout {
        ZeroLayout::None => 0,
        ZeroLayout::BlockScalar { offset_bytes, .. } => *offset_bytes,
        ZeroLayout::Hierarchical { dmin_offset, .. } => *dmin_offset,
        ZeroLayout::StaticBias { .. } => 0,
    }
}

/// 判断 format 是否有有效的 zero-point（BlockScalar 或 Hierarchical）。
fn has_dynamic_zero_point(desc: &QuantFormatDescriptor) -> bool {
    matches!(desc.zero_layout,
        ZeroLayout::BlockScalar { .. } | ZeroLayout::Hierarchical { .. }
    )
}

/// 生成 Weight 分族 trace 模板。
///
/// Weight trace: 量化 block 循环加载 → scale 加载 → zero 加载 →
/// data 解包 → 反量化 (dequant = data * scale 或 (data - zero) * scale) → FMA 累加。
///
/// 用于 QuantGemm 内层循环中 B 矩阵（量化权重）的加载和反量化。
/// 当 format_desc 可用时，利用精确的 layout 生成 scale/zero/data load 序列。
///
/// 输入 slots:
///   [0] = weight_ptr  (量化权重矩阵 B 的 base pointer)
///   [1] = output_acc  (F32 累加器)
///   [2] = activation  (F32 激活向量 A，用于 dot product)
///
/// 输出: 反量化后的 F32 权重 block → 与激活向量点积 → 累加到 output_acc。
pub fn build_weight_trace(params: &TraceTemplateParams) -> Vec<TraceOp> {
    let quant_type = params.quant_type;
    let block_size = params.block_size;
    let weight_ptr = ValueId(0);
    let acc_ptr = ValueId(1);
    let act_ptr = ValueId(2);

    let desc = params.format_desc.as_ref();
    let scale_offset = desc.map_or(0, scale_offset_from_desc);
    let data_offset = desc.map_or(0, data_offset_from_desc);
    let mut body = vec![
        TraceOp::Input(0), // [0] weight_ptr
        TraceOp::Input(1), // [1] acc_ptr
        TraceOp::Input(2), // [2] act_ptr
    ];

    // Scale load: f16 → broadcast f32
    body.push(TraceOp::QuantScaleLoad {
        source: weight_ptr,
        offset: scale_offset,
        dtype: quant_type,
    });
    let scale_val = ValueId((body.len() - 1) as u32);

    // Zero-point load (if format has dynamic zero-point)
    let zero_val = if let Some(d) = desc {
        if has_dynamic_zero_point(d) {
            body.push(TraceOp::QuantZeroLoad {
                source: weight_ptr,
                offset: zero_offset_from_desc(d),
                zp_type: d.zero_layout.clone(),
            });
            Some(ValueId((body.len() - 1) as u32))
        } else {
            None
        }
    } else {
        None
    };

    // Data load: packed data → unpack → F32
    body.push(TraceOp::QuantDataLoad {
        source: weight_ptr,
        offset: data_offset,
        quant_type,
        block_size,
    });
    let data_val = ValueId((body.len() - 1) as u32);

    // Dequantize: data * scale (or (data - zero) * scale)
    let dequant_val = if let Some(zv) = zero_val {
        // data - zero
        body.push(TraceOp::Sub(data_val, zv));
        let centered = ValueId((body.len() - 1) as u32);
        // centered * scale
        body.push(TraceOp::Mul(centered, scale_val));
        ValueId((body.len() - 1) as u32)
    } else {
        // data * scale
        body.push(TraceOp::Mul(data_val, scale_val));
        ValueId((body.len() - 1) as u32)
    };

    // Dot product with activation: dequant_val * act → accumulate
    body.push(TraceOp::Mul(dequant_val, act_ptr));
    let product = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::Add(product, acc_ptr));
    let _accumulated = ValueId((body.len() - 1) as u32);

    body
}

/// 生成 Activation 分族 trace 模板。
///
/// Activation trace: 逐 block 加载量化激活 → 反量化 → 输出 F32 中间结果。
/// 用于动态量化场景: 输入激活被量化为 INT8/INT4，在 GEMM 之前反量化。
///
/// 输入 slots:
///   [0] = activation_ptr  (量化激活 A 的 base pointer)
///   [1] = output_ptr      (F32 反量化输出 buffer)
///
/// 输出: F32 反量化后的激活值写入 output_ptr。
pub fn build_activation_trace(params: &TraceTemplateParams) -> Vec<TraceOp> {
    let quant_type = params.quant_type;
    let block_size = params.block_size;
    let act_ptr = ValueId(0);
    let out_ptr = ValueId(1);

    let desc = params.format_desc.as_ref();
    let scale_offset = desc.map_or(0, scale_offset_from_desc);
    let data_offset = desc.map_or(0, data_offset_from_desc);

    let mut body = vec![
        TraceOp::Input(0), // [0] activation_ptr
        TraceOp::Input(1), // [1] output_ptr
    ];

    // Scale load
    body.push(TraceOp::QuantScaleLoad {
        source: act_ptr,
        offset: scale_offset,
        dtype: quant_type,
    });
    let scale_val = ValueId((body.len() - 1) as u32);

    // Zero-point load
    let zero_val = if let Some(d) = desc {
        if has_dynamic_zero_point(d) {
            body.push(TraceOp::QuantZeroLoad {
                source: act_ptr,
                offset: zero_offset_from_desc(d),
                zp_type: d.zero_layout.clone(),
            });
            Some(ValueId((body.len() - 1) as u32))
        } else {
            None
        }
    } else {
        None
    };

    // Data load — auto_select handles nibble splitting via QuantFormatDescriptor lookup
    body.push(TraceOp::QuantDataLoad {
        source: act_ptr,
        offset: data_offset,
        quant_type,
        block_size,
    });
    let data_val = ValueId((body.len() - 1) as u32);

    // Dequantize
    if let Some(zv) = zero_val {
        body.push(TraceOp::Sub(data_val, zv));
        let centered = ValueId((body.len() - 1) as u32);
        body.push(TraceOp::Mul(centered, scale_val));
        let dequant = ValueId((body.len() - 1) as u32);
        // Store result to output pointer
        body.push(TraceOp::VecStoreIndexed {
            base: out_ptr,
            offset: ValueId(3), // placeholder offset value slot
            value: dequant,
        });
    } else {
        body.push(TraceOp::Mul(data_val, scale_val));
        let dequant = ValueId((body.len() - 1) as u32);
        body.push(TraceOp::VecStoreIndexed {
            base: out_ptr,
            offset: ValueId(3),
            value: dequant,
        });
    }

    body
}

/// 生成 KvCache 分族 trace 模板。
///
/// KV cache trace: 按 slot/page 加载缓存中的量化 K/V 向量 → 反量化 → F32 输出。
/// 支持跨 page boundary 的 slot 访问（paged attention）。
///
/// 输入 slots:
///   [0] = kv_ptr       (KV cache base pointer)
///   [1] = slot_index   (当前 slot/page index)
///   [2] = output_ptr   (F32 反量化输出 buffer)
///
/// 输出: F32 反量化后的 K/V 向量写入 output_ptr。
pub fn build_kv_cache_trace(params: &TraceTemplateParams) -> Vec<TraceOp> {
    let quant_type = params.quant_type;
    let block_size = params.block_size;
    let kv_ptr = ValueId(0);
    let slot_idx = ValueId(1);
    let out_ptr = ValueId(2);

    let desc = params.format_desc.as_ref();
    let scale_offset = desc.map_or(0, scale_offset_from_desc);
    let data_offset = desc.map_or(0, data_offset_from_desc);

    let mut body = vec![
        TraceOp::Input(0), // [0] kv_ptr
        TraceOp::Input(1), // [1] slot_index
        TraceOp::Input(2), // [2] output_ptr
    ];

    // Compute slot base address: kv_ptr + slot_index * stride
    // stride = per-slot bytes (quantized K or V vector size)
    let slot_stride = desc.map_or(block_size / 2, |d| d.block_bytes);
    body.push(TraceOp::Const(slot_stride as f64));
    let stride_val = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::Mul(slot_idx, stride_val));
    let byte_offset_val = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::Add(kv_ptr, byte_offset_val));
    let slot_base = ValueId((body.len() - 1) as u32);

    // Scale load from slot base
    body.push(TraceOp::QuantScaleLoad {
        source: slot_base,
        offset: scale_offset,
        dtype: quant_type,
    });
    let scale_val = ValueId((body.len() - 1) as u32);

    // Zero-point load
    let zero_val = if let Some(d) = desc {
        if has_dynamic_zero_point(d) {
            body.push(TraceOp::QuantZeroLoad {
                source: slot_base,
                offset: zero_offset_from_desc(d),
                zp_type: d.zero_layout.clone(),
            });
            Some(ValueId((body.len() - 1) as u32))
        } else {
            None
        }
    } else {
        None
    };

    // Data load from slot base — auto_select handles nibble splitting
    body.push(TraceOp::QuantDataLoad {
        source: slot_base,
        offset: data_offset,
        quant_type,
        block_size,
    });
    let data_val = ValueId((body.len() - 1) as u32);

    // Dequantize
    let dequant = if let Some(zv) = zero_val {
        body.push(TraceOp::Sub(data_val, zv));
        let centered = ValueId((body.len() - 1) as u32);
        body.push(TraceOp::Mul(centered, scale_val));
        ValueId((body.len() - 1) as u32)
    } else {
        body.push(TraceOp::Mul(data_val, scale_val));
        ValueId((body.len() - 1) as u32)
    };

    // Store to output
    body.push(TraceOp::VecStoreIndexed {
        base: out_ptr,
        offset: ValueId(3), // placeholder offset slot
        value: dequant,
    });

    body
}

/// 生成 Rope 分族 trace 模板。
///
/// RoPE trace: 加载 position index → cos/sin lookup → 对 (x, y) 对执行
/// rotate_half: out = x * cos - y * sin, out_next = x * sin + y * cos。
///
/// 输入 slots:
///   [0] = input_ptr    (F32 输入张量 [seq_len, head_dim])
///   [1] = cos_ptr      (F32 cos lookup table)
///   [2] = sin_ptr      (F32 sin lookup table)
///   [3] = pos_ptr      (i32 position indices)
///   [4] = output_ptr   (F32 输出张量)
///
/// 输出: RoPE 旋转后的 F32 输出写入 output_ptr。
pub fn build_rope_trace(params: &TraceTemplateParams) -> Vec<TraceOp> {
    let block_size = params.block_size;
    let input_ptr = ValueId(0);
    let cos_ptr = ValueId(1);
    let sin_ptr = ValueId(2);
    let _pos_ptr = ValueId(3);
    let out_ptr = ValueId(4);

    let half_dim = block_size / 2;

    let mut body = vec![
        TraceOp::Input(0), // [0] input_ptr
        TraceOp::Input(1), // [1] cos_ptr
        TraceOp::Input(2), // [2] sin_ptr
        TraceOp::Input(3), // [3] pos_ptr
        TraceOp::Input(4), // [4] output_ptr
    ];

    // Load x (first half of head_dim)
    body.push(TraceOp::Const(0.0));
    let zero_offset = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::VecLoadIndexed {
        base: input_ptr,
        offset: zero_offset,
    });
    let x_val = ValueId((body.len() - 1) as u32);

    // Load y (second half of head_dim)
    body.push(TraceOp::Const(half_dim as f64 * 4.0)); // byte offset for second half
    let half_offset = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::VecLoadIndexed {
        base: input_ptr,
        offset: half_offset,
    });
    let y_val = ValueId((body.len() - 1) as u32);

    // Load cos(position, head_dim//2)
    body.push(TraceOp::Const(0.0));
    let cos_offset = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::VecLoadIndexed {
        base: cos_ptr,
        offset: cos_offset,
    });
    let cos_val = ValueId((body.len() - 1) as u32);

    // Load sin(position, head_dim//2)
    body.push(TraceOp::Const(0.0));
    let sin_offset = ValueId((body.len() - 1) as u32);
    body.push(TraceOp::VecLoadIndexed {
        base: sin_ptr,
        offset: sin_offset,
    });
    let sin_val = ValueId((body.len() - 1) as u32);

    // rotate_half: x * cos
    body.push(TraceOp::Mul(x_val, cos_val));
    let x_cos = ValueId((body.len() - 1) as u32);

    // y * sin
    body.push(TraceOp::Mul(y_val, sin_val));
    let y_sin = ValueId((body.len() - 1) as u32);

    // x * sin
    body.push(TraceOp::Mul(x_val, sin_val));
    let x_sin = ValueId((body.len() - 1) as u32);

    // y * cos
    body.push(TraceOp::Mul(y_val, cos_val));
    let y_cos = ValueId((body.len() - 1) as u32);

    // out_first_half = x * cos - y * sin
    body.push(TraceOp::Sub(x_cos, y_sin));
    let out_first = ValueId((body.len() - 1) as u32);

    // out_second_half = x * sin + y * cos
    body.push(TraceOp::Add(x_sin, y_cos));
    let out_second = ValueId((body.len() - 1) as u32);

    // Store first half
    body.push(TraceOp::VecStoreIndexed {
        base: out_ptr,
        offset: zero_offset,
        value: out_first,
    });

    // Store second half
    body.push(TraceOp::VecStoreIndexed {
        base: out_ptr,
        offset: half_offset,
        value: out_second,
    });

    body
}

/// 根据 DataKind 分发到对应的 trace 模板生成函数。
///
/// 这是统一入口点，根据 params.data_kind 选择:
/// - Weight → build_weight_trace
/// - Activation → build_activation_trace
/// - KvCache → build_kv_cache_trace
/// - Rope → build_rope_trace
pub fn build_data_kind_trace(params: &TraceTemplateParams) -> Vec<TraceOp> {
    match params.data_kind {
        DataKind::Weight => build_weight_trace(params),
        DataKind::Activation => build_activation_trace(params),
        DataKind::KvCache => build_kv_cache_trace(params),
        DataKind::Rope => build_rope_trace(params),
    }
}

/// 便捷构造器: 从 QuantType + block_size 创建 Weight 分族 trace。
pub fn build_weight_trace_simple(
    quant_type: QuantType,
    block_size: usize,
    dtype: QuantOutputDtype,
) -> Vec<TraceOp> {
    build_data_kind_trace(&TraceTemplateParams {
        data_kind: DataKind::Weight,
        quant_type,
        block_size,
        dtype,
        format_desc: None,
    })
}

/// 便捷构造器: 从 QuantType + block_size 创建 Activation 分族 trace。
pub fn build_activation_trace_simple(
    quant_type: QuantType,
    block_size: usize,
    dtype: QuantOutputDtype,
) -> Vec<TraceOp> {
    build_data_kind_trace(&TraceTemplateParams {
        data_kind: DataKind::Activation,
        quant_type,
        block_size,
        dtype,
        format_desc: None,
    })
}

/// 便捷构造器: 从 QuantType + block_size 创建 KvCache 分族 trace。
pub fn build_kv_cache_trace_simple(
    quant_type: QuantType,
    block_size: usize,
    dtype: QuantOutputDtype,
) -> Vec<TraceOp> {
    build_data_kind_trace(&TraceTemplateParams {
        data_kind: DataKind::KvCache,
        quant_type,
        block_size,
        dtype,
        format_desc: None,
    })
}

/// 便捷构造器: 从 block_size (head_dim) 创建 Rope 分族 trace。
pub fn build_rope_trace_simple(
    block_size: usize,
    dtype: QuantOutputDtype,
) -> Vec<TraceOp> {
    build_data_kind_trace(&TraceTemplateParams {
        data_kind: DataKind::Rope,
        quant_type: QuantType::F32,
        block_size,
        dtype,
        format_desc: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_trace_has_valid_ssa() {
        let body = build_weight_trace_simple(QuantType::Q4_0, 32, QuantOutputDtype::F32);
        assert!(body.len() > 4, "weight trace should have multiple ops, got {}", body.len());
        // Verify all value references are within bounds
        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::QuantScaleLoad { .. } | TraceOp::QuantZeroLoad { .. }
                | TraceOp::QuantDataLoad { .. } => {}
                TraceOp::Neg(a) | TraceOp::Exp(a) | TraceOp::Sqrt(a) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}: operand {a:?}");
                }
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b) | TraceOp::Div(a, b) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}: left operand {a:?}");
                    assert!((b.0 as usize) < i, "SSA violation at {i}: right operand {b:?}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn activation_trace_has_valid_ssa() {
        let body = build_activation_trace_simple(QuantType::Q8_0, 32, QuantOutputDtype::F32);
        assert!(body.len() > 4, "activation trace should have multiple ops, got {}", body.len());
        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::QuantScaleLoad { .. } | TraceOp::QuantZeroLoad { .. }
                | TraceOp::QuantDataLoad { .. } => {}
                TraceOp::VecStoreIndexed { base, offset, value } => {
                    assert!((base.0 as usize) < i, "SSA violation at {i}: base {base:?}");
                    assert!((offset.0 as usize) < i, "SSA violation at {i}: offset {offset:?}");
                    assert!((value.0 as usize) < i, "SSA violation at {i}: value {value:?}");
                }
                TraceOp::Add(a, b) | TraceOp::Sub(a, b) | TraceOp::Mul(a, b) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}");
                    assert!((b.0 as usize) < i, "SSA violation at {i}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn kv_cache_trace_has_valid_ssa() {
        let body = build_kv_cache_trace_simple(QuantType::Q4_0, 32, QuantOutputDtype::F32);
        assert!(body.len() > 4, "kv_cache trace should have multiple ops, got {}", body.len());
        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::VecLoadIndexed { base, offset } => {
                    assert!((base.0 as usize) < i, "SSA violation at {i}");
                    assert!((offset.0 as usize) < i, "SSA violation at {i}");
                }
                TraceOp::VecStoreIndexed { base, offset, value } => {
                    assert!((base.0 as usize) < i, "SSA violation at {i}");
                    assert!((offset.0 as usize) < i, "SSA violation at {i}");
                    assert!((value.0 as usize) < i, "SSA violation at {i}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn rope_trace_has_valid_ssa() {
        let body = build_rope_trace_simple(64, QuantOutputDtype::F32);
        assert!(body.len() > 10, "rope trace should have many ops, got {}", body.len());
        // Verify all binary ops have valid SSA refs
        for (i, op) in body.iter().enumerate() {
            match op {
                TraceOp::Input(_) | TraceOp::Const(_) => {}
                TraceOp::Mul(a, b) | TraceOp::Add(a, b) | TraceOp::Sub(a, b) => {
                    assert!((a.0 as usize) < i, "SSA violation at {i}: left {a:?}");
                    assert!((b.0 as usize) < i, "SSA violation at {i}: right {b:?}");
                }
                TraceOp::VecLoadIndexed { base, offset } => {
                    assert!((base.0 as usize) < i, "SSA violation at {i}: base");
                    assert!((offset.0 as usize) < i, "SSA violation at {i}: offset");
                }
                TraceOp::VecStoreIndexed { base, offset, value } => {
                    assert!((base.0 as usize) < i, "SSA violation at {i}: base");
                    assert!((offset.0 as usize) < i, "SSA violation at {i}: offset");
                    assert!((value.0 as usize) < i, "SSA violation at {i}: value");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn data_kind_dispatch_matches_specialized() {
        // Verify dispatch returns same result as direct call for each kind
        let params = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::Q4_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };
        let dispatched = build_data_kind_trace(&params);
        let direct = build_weight_trace(&params);
        assert_eq!(dispatched.len(), direct.len(),
            "dispatch should match direct call for Weight");
    }

    // ── 13 new tests below ──

    #[test]
    fn data_kind_all_variants_are_distinct() {
        let kinds = [DataKind::Weight, DataKind::Activation, DataKind::KvCache, DataKind::Rope];
        for i in 0..kinds.len() {
            for j in (i + 1)..kinds.len() {
                assert_ne!(kinds[i], kinds[j],
                    "DataKind variants at index {i} and {j} must be distinct");
            }
        }
    }

    #[test]
    fn quant_output_dtype_elem_bytes_covers_all_variants() {
        assert_eq!(QuantOutputDtype::F32.elem_bytes(), 4, "F32 must be 4 bytes");
        assert_eq!(QuantOutputDtype::F16.elem_bytes(), 2, "F16 must be 2 bytes");
        assert_eq!(QuantOutputDtype::BF16.elem_bytes(), 2, "BF16 must be 2 bytes");
    }

    #[test]
    fn trace_template_params_clone_preserves_fields() {
        let desc = QuantFormatDescriptor::for_type(QuantType::Q4_0);
        let original = TraceTemplateParams {
            data_kind: DataKind::KvCache,
            quant_type: QuantType::Q4_0,
            block_size: 64,
            dtype: QuantOutputDtype::BF16,
            format_desc: Some(desc.clone()),
        };
        let cloned = original.clone();
        assert_eq!(cloned.data_kind, original.data_kind);
        assert_eq!(cloned.quant_type, original.quant_type);
        assert_eq!(cloned.block_size, original.block_size);
        assert_eq!(cloned.dtype, original.dtype);
        assert!(cloned.format_desc.is_some());
        assert_eq!(cloned.format_desc.as_ref().unwrap().name, original.format_desc.as_ref().unwrap().name);
    }

    #[test]
    fn trace_template_params_struct_update_syntax() {
        let base = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::Q8_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };
        let derived = TraceTemplateParams {
            data_kind: DataKind::Activation,
            block_size: 128,
            ..base
        };
        assert_eq!(derived.data_kind, DataKind::Activation);
        assert_eq!(derived.quant_type, QuantType::Q8_0);
        assert_eq!(derived.block_size, 128);
        assert_eq!(derived.dtype, QuantOutputDtype::F32);
        assert!(derived.format_desc.is_none());
    }

    #[test]
    fn trace_template_params_format_desc_none_is_valid() {
        let params = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::Q4_1,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };
        let body = build_weight_trace(&params);
        assert!(!body.is_empty(), "trace body must not be empty even without format_desc");
        assert!(matches!(body[0], TraceOp::Input(0)), "first op must be Input(0)");
    }

    #[test]
    fn build_data_kind_trace_dispatches_all_four_kinds() {
        let kinds = [DataKind::Weight, DataKind::Activation, DataKind::KvCache, DataKind::Rope];
        let mut lengths = Vec::new();
        for kind in kinds {
            let params = TraceTemplateParams {
                data_kind: kind,
                quant_type: QuantType::Q4_0,
                block_size: 32,
                dtype: QuantOutputDtype::F32,
                format_desc: None,
            };
            let body = build_data_kind_trace(&params);
            assert!(!body.is_empty(), "data_kind={kind:?} produced empty trace");
            lengths.push(body.len());
        }
        // Rope has the most ops; all should differ or at least not be empty
        assert!(lengths.iter().all(|&l| l > 0), "all dispatch results must be non-empty");
    }

    #[test]
    fn weight_trace_starts_with_three_inputs() {
        let body = build_weight_trace_simple(QuantType::Q5_0, 32, QuantOutputDtype::F32);
        assert!(body.len() >= 3, "weight trace must have at least 3 input ops");
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Input(2)));
    }

    #[test]
    fn activation_trace_starts_with_two_inputs() {
        let body = build_activation_trace_simple(QuantType::Q4_0, 32, QuantOutputDtype::F16);
        assert!(body.len() >= 2, "activation trace must have at least 2 input ops");
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
    }

    #[test]
    fn kv_cache_trace_starts_with_three_inputs() {
        let body = build_kv_cache_trace_simple(QuantType::Q8_0, 32, QuantOutputDtype::BF16);
        assert!(body.len() >= 3, "kv_cache trace must have at least 3 input ops");
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Input(2)));
    }

    #[test]
    fn rope_trace_starts_with_five_inputs() {
        let body = build_rope_trace_simple(64, QuantOutputDtype::F32);
        assert!(body.len() >= 5, "rope trace must have at least 5 input ops");
        assert!(matches!(body[0], TraceOp::Input(0)));
        assert!(matches!(body[1], TraceOp::Input(1)));
        assert!(matches!(body[2], TraceOp::Input(2)));
        assert!(matches!(body[3], TraceOp::Input(3)));
        assert!(matches!(body[4], TraceOp::Input(4)));
    }

    #[test]
    fn rope_trace_half_dim_offset_is_correct() {
        let block_size: usize = 128;
        let body = build_rope_trace_simple(block_size, QuantOutputDtype::F32);
        let half_dim = block_size / 2;
        // The second VecLoadIndexed should use byte_offset = half_dim * 4 (f32 bytes)
        let expected_half_byte_offset = half_dim as f64 * 4.0;
        let mut found_half_offset = false;
        for op in &body {
            if let TraceOp::Const(val) = op {
                if (*val - expected_half_byte_offset).abs() < f64::EPSILON {
                    found_half_offset = true;
                }
            }
        }
        assert!(found_half_offset,
            "rope trace must contain Const({expected_half_byte_offset}) for half-dim byte offset");
    }

    #[test]
    fn block_size_one_does_not_panic() {
        let result = std::panic::catch_unwind(|| {
            let params = TraceTemplateParams {
                data_kind: DataKind::Weight,
                quant_type: QuantType::Q4_0,
                block_size: 1,
                dtype: QuantOutputDtype::F32,
                format_desc: None,
            };
            let _ = build_data_kind_trace(&params);
        });
        assert!(result.is_ok(), "block_size=1 must not panic");
    }

    #[test]
    fn data_kind_debug_format_contains_variant_name() {
        let debug_str = format!("{:?}", DataKind::KvCache);
        assert!(debug_str.contains("KvCache"), "Debug output must contain variant name: got {debug_str}");
        let debug_str = format!("{:?}", DataKind::Rope);
        assert!(debug_str.contains("Rope"), "Debug output must contain variant name: got {debug_str}");
    }

    // ── 10 additional tests ──

    #[test]
    fn weight_trace_contains_quant_scale_load() {
        // Arrange
        let params = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::Q4_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };

        // Act
        let body = build_weight_trace(&params);

        // Assert: body must contain a QuantScaleLoad op
        let has_scale_load = body.iter().any(|op| matches!(op, TraceOp::QuantScaleLoad { .. }));
        assert!(has_scale_load, "weight trace must contain QuantScaleLoad");
    }

    #[test]
    fn weight_trace_contains_quant_data_load() {
        // Arrange
        let params = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::Q8_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };

        // Act
        let body = build_weight_trace(&params);

        // Assert: body must contain a QuantDataLoad with matching quant_type and block_size
        let found = body.iter().any(|op| {
            matches!(op, TraceOp::QuantDataLoad {
                quant_type: QuantType::Q8_0,
                block_size: 32,
                ..
            })
        });
        assert!(found, "weight trace must contain QuantDataLoad with Q8_0 block_size=32");
    }

    #[test]
    fn weight_trace_with_format_desc_includes_zero_load() {
        // Arrange: AWQ4 has ZeroLayout::BlockScalar — a dynamic zero-point
        let desc = QuantFormatDescriptor::for_type(QuantType::AWQ4);
        let params = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::AWQ4,
            block_size: 128,
            dtype: QuantOutputDtype::F32,
            format_desc: Some(desc),
        };

        // Act
        let body = build_weight_trace(&params);

        // Assert: body must contain QuantZeroLoad because AWQ4 has BlockScalar zero-point
        let has_zero_load = body.iter().any(|op| matches!(op, TraceOp::QuantZeroLoad { .. }));
        assert!(has_zero_load, "AWQ4 weight trace must contain QuantZeroLoad for dynamic zero-point");
    }

    #[test]
    fn kv_cache_trace_slot_stride_uses_block_bytes() {
        // Arrange: provide a format_desc with known block_bytes
        let desc = QuantFormatDescriptor::for_type(QuantType::Q4_0);
        let block_bytes = desc.block_bytes;
        let params = TraceTemplateParams {
            data_kind: DataKind::KvCache,
            quant_type: QuantType::Q4_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: Some(desc),
        };

        // Act
        let body = build_kv_cache_trace(&params);

        // Assert: body must contain Const(block_bytes) as the slot stride
        let has_stride_const = body.iter().any(|op| {
            matches!(op, TraceOp::Const(v) if (*v - block_bytes as f64).abs() < f64::EPSILON)
        });
        assert!(has_stride_const,
            "kv_cache trace with format_desc must contain Const({block_bytes}) as slot stride");
    }

    #[test]
    fn rope_trace_contains_rotate_half_mul_sub_add_pattern() {
        // Arrange
        let body = build_rope_trace_simple(64, QuantOutputDtype::F32);

        // Act: count Mul, Sub, Add ops — RoPE uses 4 Mul + 1 Sub + 1 Add for rotate_half
        let mul_count = body.iter().filter(|op| matches!(op, TraceOp::Mul(_, _))).count();
        let sub_count = body.iter().filter(|op| matches!(op, TraceOp::Sub(_, _))).count();
        let add_count = body.iter().filter(|op| matches!(op, TraceOp::Add(_, _))).count();

        // Assert: 4 Mul (x*cos, y*sin, x*sin, y*cos), 1 Sub (x*cos - y*sin), 1 Add (x*sin + y*cos)
        assert_eq!(mul_count, 4, "rope trace must have exactly 4 Mul ops for rotate_half, got {mul_count}");
        assert_eq!(sub_count, 1, "rope trace must have exactly 1 Sub op for out_first_half, got {sub_count}");
        assert_eq!(add_count, 1, "rope trace must have exactly 1 Add op for out_second_half, got {add_count}");
    }

    #[test]
    fn rope_trace_has_two_vec_store_indexed() {
        // Arrange
        let body = build_rope_trace_simple(128, QuantOutputDtype::F32);

        // Act
        let store_count = body.iter().filter(|op| matches!(op, TraceOp::VecStoreIndexed { .. })).count();

        // Assert: RoPE stores two halves (first half and second half)
        assert_eq!(store_count, 2, "rope trace must store first and second half, got {store_count}");
    }

    #[test]
    fn quant_output_dtype_equality_and_distinction() {
        // Arrange
        let f32_a = QuantOutputDtype::F32;
        let f32_b = QuantOutputDtype::F32;
        let f16 = QuantOutputDtype::F16;
        let bf16 = QuantOutputDtype::BF16;

        // Assert
        assert_eq!(f32_a, f32_b, "same variants must be equal");
        assert_ne!(f32_a, f16, "F32 and F16 must be distinct");
        assert_ne!(f32_a, bf16, "F32 and BF16 must be distinct");
        assert_ne!(f16, bf16, "F16 and BF16 must be distinct");
    }

    #[test]
    fn value_id_none_is_max_u32() {
        // Arrange
        let none = ValueId::NONE;

        // Assert
        assert_eq!(none.0, u32::MAX, "ValueId::NONE must be u32::MAX");
        assert!(!none.is_some(), "ValueId::NONE.is_some() must return false");
    }

    #[test]
    fn value_id_saturating_sub_does_not_underflow() {
        // Arrange
        let id = ValueId(0);

        // Act
        let result = id.saturating_sub(5);

        // Assert: saturating_sub must not underflow below 0
        assert_eq!(result.0, 0, "ValueId(0).saturating_sub(5) must be ValueId(0), not underflow");
    }

    #[test]
    fn value_id_sub_tracked_by_difference() {
        // Arrange
        let id = ValueId(10);

        // Act
        let result = id - 3;

        // Assert
        assert_eq!(result.0, 7, "ValueId(10) - 3 must be ValueId(7)");
    }

    #[test]
    fn build_data_kind_trace_produces_different_lengths_per_kind() {
        // Arrange: use the same quant_type/block_size for all kinds
        let kinds = [DataKind::Weight, DataKind::Activation, DataKind::KvCache, DataKind::Rope];

        // Act
        let lengths: Vec<usize> = kinds.iter().map(|kind| {
            let params = TraceTemplateParams {
                data_kind: *kind,
                quant_type: QuantType::Q4_0,
                block_size: 32,
                dtype: QuantOutputDtype::F32,
                format_desc: None,
            };
            build_data_kind_trace(&params).len()
        }).collect();

        // Assert: Rope should have the most ops due to 5 inputs + 4 loads + 4 mul + sub + add + 2 stores
        let rope_idx = kinds.iter().position(|k| *k == DataKind::Rope).unwrap();
        for (i, &len) in lengths.iter().enumerate() {
            if i != rope_idx {
                assert!(lengths[rope_idx] > len,
                    "Rope trace ({} ops) should be longer than {:?} trace ({} ops)",
                    lengths[rope_idx], kinds[i], len);
            }
        }
    }

    #[test]
    fn activation_trace_contains_vec_store_indexed() {
        // Arrange
        let params = TraceTemplateParams {
            data_kind: DataKind::Activation,
            quant_type: QuantType::Q4_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };

        // Act
        let body = build_activation_trace(&params);

        // Assert: activation trace must write result via VecStoreIndexed
        let has_store = body.iter().any(|op| matches!(op, TraceOp::VecStoreIndexed { .. }));
        assert!(has_store, "activation trace must contain VecStoreIndexed for output write");
    }

    // ── 10 more tests (total 40) ──

    #[test]
    fn value_id_is_some_returns_true_for_valid_ids() {
        // Arrange
        let valid = ValueId(0);
        let mid = ValueId(100);
        let almost_none = ValueId(u32::MAX - 1);

        // Assert: all non-MAX ValueIds must report is_some() == true
        assert!(valid.is_some(), "ValueId(0) must be some");
        assert!(mid.is_some(), "ValueId(100) must be some");
        assert!(almost_none.is_some(), "ValueId(u32::MAX - 1) must be some");
        assert!(!ValueId::NONE.is_some(), "ValueId::NONE must not be some");
    }

    #[test]
    fn value_id_display_shows_v_prefix() {
        // Arrange
        let id = ValueId(42);

        // Act
        let display = format!("{id}");

        // Assert
        assert_eq!(display, "v42", "ValueId(42) must display as v42");
    }

    #[test]
    fn weight_trace_ends_with_add_accumulator() {
        // Arrange: build a weight trace without format_desc
        let body = build_weight_trace_simple(QuantType::Q4_0, 32, QuantOutputDtype::F32);

        // Act: the final op should be an Add accumulating into the accumulator
        let last = body.last().expect("weight trace must not be empty");

        // Assert: the trace must end with Add (product + acc_ptr)
        assert!(matches!(last, TraceOp::Add(_, _)),
            "weight trace must end with Add for accumulation, got {last:?}");
    }

    #[test]
    fn weight_trace_without_format_desc_has_no_zero_load() {
        // Arrange: Q4_0 without format_desc — zero detection not possible
        let params = TraceTemplateParams {
            data_kind: DataKind::Weight,
            quant_type: QuantType::Q4_0,
            block_size: 32,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };

        // Act
        let body = build_weight_trace(&params);

        // Assert: without format_desc, no QuantZeroLoad should be emitted
        let has_zero = body.iter().any(|op| matches!(op, TraceOp::QuantZeroLoad { .. }));
        assert!(!has_zero,
            "weight trace without format_desc must not contain QuantZeroLoad");
    }

    #[test]
    fn activation_trace_with_format_desc_includes_zero_load() {
        // Arrange: GPTQ4 has ZeroLayout::BlockScalar — dynamic zero-point
        let desc = QuantFormatDescriptor::for_type(QuantType::GPTQ4);
        let params = TraceTemplateParams {
            data_kind: DataKind::Activation,
            quant_type: QuantType::GPTQ4,
            block_size: 128,
            dtype: QuantOutputDtype::F32,
            format_desc: Some(desc),
        };

        // Act
        let body = build_activation_trace(&params);

        // Assert: GPTQ4 activation trace must contain QuantZeroLoad
        let has_zero_load = body.iter().any(|op| matches!(op, TraceOp::QuantZeroLoad { .. }));
        assert!(has_zero_load,
            "GPTQ4 activation trace with format_desc must contain QuantZeroLoad");
    }

    #[test]
    fn kv_cache_trace_with_format_desc_includes_zero_load() {
        // Arrange: Q4K has ZeroLayout::Hierarchical — a dynamic zero-point
        let desc = QuantFormatDescriptor::for_type(QuantType::Q4K);
        let params = TraceTemplateParams {
            data_kind: DataKind::KvCache,
            quant_type: QuantType::Q4K,
            block_size: desc.block_size,
            dtype: QuantOutputDtype::F32,
            format_desc: Some(desc.clone()),
        };

        // Act
        let body = build_kv_cache_trace(&params);

        // Assert: Q4K KV cache trace must contain QuantZeroLoad
        let has_zero_load = body.iter().any(|op| matches!(op, TraceOp::QuantZeroLoad { .. }));
        assert!(has_zero_load,
            "Q4K kv_cache trace with format_desc must contain QuantZeroLoad");
    }

    #[test]
    fn kv_cache_trace_without_format_desc_uses_block_size_half_as_stride() {
        // Arrange: block_size=64, no format_desc → stride = 64/2 = 32
        let params = TraceTemplateParams {
            data_kind: DataKind::KvCache,
            quant_type: QuantType::Q4_0,
            block_size: 64,
            dtype: QuantOutputDtype::F32,
            format_desc: None,
        };

        // Act
        let body = build_kv_cache_trace(&params);

        // Assert: stride must be block_size / 2 = 32
        let expected_stride = 32.0_f64;
        let has_stride = body.iter().any(|op| {
            matches!(op, TraceOp::Const(v) if (*v - expected_stride).abs() < f64::EPSILON)
        });
        assert!(has_stride,
            "kv_cache trace without format_desc must use block_size/2 = {expected_stride} as stride");
    }

    #[test]
    fn kv_cache_trace_ends_with_vec_store_indexed() {
        // Arrange
        let body = build_kv_cache_trace_simple(QuantType::Q8_0, 32, QuantOutputDtype::F32);

        // Act
        let last = body.last().expect("kv_cache trace must not be empty");

        // Assert: KV cache trace must end with VecStoreIndexed writing dequantized output
        assert!(matches!(last, TraceOp::VecStoreIndexed { .. }),
            "kv_cache trace must end with VecStoreIndexed, got {last:?}");
    }

    #[test]
    fn weight_trace_different_quant_types_produce_different_scale_loads() {
        // Arrange: two different quant types
        let body_q4 = build_weight_trace_simple(QuantType::Q4_0, 32, QuantOutputDtype::F32);
        let body_q8 = build_weight_trace_simple(QuantType::Q8_0, 32, QuantOutputDtype::F32);

        // Act: extract QuantScaleLoad dtype fields
        let scale_q4 = body_q4.iter().find_map(|op| {
            if let TraceOp::QuantScaleLoad { dtype, .. } = op { Some(*dtype) } else { None }
        });
        let scale_q8 = body_q8.iter().find_map(|op| {
            if let TraceOp::QuantScaleLoad { dtype, .. } = op { Some(*dtype) } else { None }
        });

        // Assert: both must have QuantScaleLoad with their respective quant_type
        assert!(scale_q4.is_some(), "Q4_0 weight trace must have QuantScaleLoad");
        assert!(scale_q8.is_some(), "Q8_0 weight trace must have QuantScaleLoad");
        assert_eq!(scale_q4.unwrap(), QuantType::Q4_0);
        assert_eq!(scale_q8.unwrap(), QuantType::Q8_0);
    }

    #[test]
    fn rope_trace_block_size_affects_half_offset_value() {
        // Arrange: use two different block sizes
        let body_64 = build_rope_trace_simple(64, QuantOutputDtype::F32);
        let body_128 = build_rope_trace_simple(128, QuantOutputDtype::F32);

        // Act: find the half-dim byte offset Const values (half_dim * 4.0)
        let find_half_offset = |body: &[TraceOp]| -> Option<f64> {
            body.iter().find_map(|op| {
                if let TraceOp::Const(v) = op {
                    // half offset = (block_size/2) * 4.0, must be > 0 and divisible by 4
                    if *v > 0.0 && (*v % 4.0).abs() < f64::EPSILON {
                        return Some(*v);
                    }
                }
                None
            })
        };

        let offset_64 = find_half_offset(&body_64);
        let offset_128 = find_half_offset(&body_128);

        // Assert: larger block_size must produce larger half offset
        assert!(offset_64.is_some(), "rope trace block_size=64 must have half-dim offset");
        assert!(offset_128.is_some(), "rope trace block_size=128 must have half-dim offset");
        assert!(offset_128.unwrap() > offset_64.unwrap(),
            "block_size=128 half offset ({}) must be > block_size=64 half offset ({})",
            offset_128.unwrap(), offset_64.unwrap());
    }
}
