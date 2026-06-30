//! 编译时数值模拟器 (REQ-LC-011)
//!
//! 对 DecodeTraceBuilder 生成的 TraceOp 序列进行标量模拟，
//! 验证量化解码的正确性，确保输出无 NaN/Inf。
//!
//! ## 功能
//! - 小规模模拟 (seq_len=4, hidden=8)
//! - 标量参考实现对比
//! - 浮点精度容差 ±1e-5
//! - 检测 NaN/Inf 输出
//!
//! SPEC: gllm-kernels/SPEC/25-JIT-LIFECYCLE-INFRASTRUCTURE.md REQ-LC-011

use crate::compiler::trace::{
    TraceOp, ValueId, ReduceKind, CmpOp, Fp8Format, ScaleSelector,
};
use crate::quant_format::{
    PackedScaleAlgorithm, QuantFormatDescriptor, QuantDataKind, ZeroLayout,
};
use crate::types::CompilerError;
use std::collections::HashMap;

/// 模拟器的值表示。
///
/// 支持整数（用于地址/索引）和浮点数（用于计算）。
#[derive(Debug, Clone, Copy, PartialEq)]
enum SimValue {
    /// 浮点数值 (f32)
    Float(f32),
    /// 整数值 (i64，用于地址计算和索引)
    Integer(i64),
    /// 未初始化/无效值
    Invalid,
}

impl SimValue {
    /// 提取浮点值，如果不是浮点则返回错误。
    fn as_float(self) -> Result<f32, CompilerError> {
        match self {
            SimValue::Float(f) => Ok(f),
            SimValue::Integer(i) => {
                // 整数可以安全地转换为浮点
                Ok(i as f32)
            }
            SimValue::Invalid => Err(CompilerError::CodegenViolation(
                "Simulation error: use of invalid value".to_string()
            )),
        }
    }

    /// 提取整数值，如果不是整数则返回错误。
    fn as_integer(self) -> Result<i64, CompilerError> {
        match self {
            SimValue::Integer(i) => Ok(i),
            SimValue::Float(f) => {
                // 浮点转整数需要检查是否为整数
                if f.fract() == 0.0 && f >= i32::MIN as f32 && f <= i32::MAX as f32 {
                    Ok(f as i64)
                } else {
                    Err(CompilerError::CodegenViolation(
                        format!("Simulation error: cannot convert non-integer float {} to integer", f)
                    ))
                }
            }
            SimValue::Invalid => Err(CompilerError::CodegenViolation(
                "Simulation error: use of invalid value".to_string()
            )),
        }
    }

    /// 检查值是否有效（非 Invalid）。
    fn is_valid(self) -> bool {
        !matches!(self, SimValue::Invalid)
    }
}

/// 标量模拟器状态。
struct SimState {
    /// ValueId → SimValue 映射
    values: HashMap<ValueId, SimValue>,
    /// 模拟的内存 (简化版: 字节数组)
    memory: Vec<u8>,
    /// 下一个分配的 ValueId（独立于 HashMap len，防止与 Input slot 碰撞）
    next_id: u32,
}

impl SimState {
    /// 创建新的模拟器状态。
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            memory: vec![0u8; 65536], // 64KB 模拟内存
            next_id: 0,
        }
    }

    /// 分配新的 ValueId（自增，不依赖 HashMap len）。
    fn alloc_id(&mut self) -> ValueId {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        id
    }

    /// 设置输入值（必须先调用 alloc_input_slots 预留空间）。
    fn set_input(&mut self, input_idx: u32, value: SimValue) {
        let id = ValueId(input_idx);
        self.values.insert(id, value);
    }

    /// 获取值，如果不存在则返回 Invalid。
    fn get(&self, id: ValueId) -> SimValue {
        self.values.get(&id).copied().unwrap_or(SimValue::Invalid)
    }

    /// 设置值。
    fn set(&mut self, id: ValueId, value: SimValue) {
        self.values.insert(id, value);
    }

    /// 从内存加载字节 (返回整数).
    fn load_u8(&self, addr: i64) -> Result<u8, CompilerError> {
        if addr < 0 || addr >= self.memory.len() as i64 {
            return Err(CompilerError::CodegenViolation(
                format!("Simulation error: out-of-bounds memory access at {}", addr)
            ));
        }
        Ok(self.memory[addr as usize])
    }

    /// 从内存加载 i16.
    fn load_i16(&self, addr: i64) -> Result<i16, CompilerError> {
        if addr < 0 || addr + 1 >= self.memory.len() as i64 {
            return Err(CompilerError::CodegenViolation(
                format!("Simulation error: out-of-bounds memory access at {}", addr)
            ));
        }
        let bytes = [self.memory[addr as usize], self.memory[addr as usize + 1]];
        Ok(i16::from_le_bytes(bytes))
    }

    /// 从内存加载 f16 (转换为 f32).
    fn load_f16(&self, addr: i64) -> Result<f32, CompilerError> {
        if addr < 0 || addr + 1 >= self.memory.len() as i64 {
            return Err(CompilerError::CodegenViolation(
                format!("Simulation error: out-of-bounds memory access at {}", addr)
            ));
        }
        let bytes = [self.memory[addr as usize], self.memory[addr as usize + 1]];
        let u = u16::from_le_bytes(bytes);
        // 简化的 f16 转 f32
        let sign = if u & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((u & 0x7C00) >> 10) as i32 - 15 + 127; // 转换到 f32 指数偏移
        let mant = (u & 0x03FF) as f32;
        if exp == 0 {
            // Subnormal or zero
            Ok(sign * mant * (2.0_f32).powi(-14 - 10))
        } else if exp == 255 - 15 + 127 {
            // Inf or NaN
            if mant == 0.0 {
                Ok(sign * f32::INFINITY)
            } else {
                Ok(f32::NAN)
            }
        } else {
            Ok(sign * (1.0 + mant / 1024.0) * (2.0_f32).powi(exp - 127))
        }
    }

    /// 存储字节到内存。
    fn store_u8(&mut self, addr: i64, value: u8) -> Result<(), CompilerError> {
        if addr < 0 || addr >= self.memory.len() as i64 {
            return Err(CompilerError::CodegenViolation(
                format!("Simulation error: out-of-bounds memory access at {}", addr)
            ));
        }
        self.memory[addr as usize] = value;
        Ok(())
    }
}

/// 模拟结果。
#[derive(Debug, Clone)]
pub struct SimResult {
    /// 输出值 (f32 向量，每个 lane 一个值)
    pub outputs: Vec<f32>,
    /// 是否包含 NaN
    pub has_nan: bool,
    /// 是否包含 Inf
    pub has_inf: bool,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// § Helper: E2M1 标量解码
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 解码单个 E2M1 4-bit 值为 f32。
///
/// E2M1 格式: 1 sign + 2 exponent (bias=1) + 1 mantissa bit。
/// 编码值范围: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
fn decode_e2m1_scalar(nibble: u8) -> f32 {
    let val = nibble & 0x0F;
    let sign = if val & 0x08 != 0 { -1.0 } else { 1.0 };
    let exp = (val >> 1) & 0x03;
    let mant = (val & 0x01) as f32;

    if exp == 0 {
        // E2M1: exponent 0 → value = 0 (subnormal 不表示)
        0.0
    } else {
        // value = (1 + mant/2) * 2^(exp - 1)
        let significand = 1.0 + mant / 2.0;
        sign * significand * (2.0_f32).powi((exp as i32) - 1)
    }
}

/// 解码 FP8 E4M3 值为 f32。
///
/// E4M3 格式: 1 sign + 4 exponent (bias=7) + 3 mantissa bits。
fn decode_fp8_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as f32;

    if exp == 0 {
        // Subnormal: value = mant / 8 * 2^(-6)
        if mant == 0.0 {
            0.0
        } else {
            sign * (mant / 8.0) * (2.0_f32).powi(-6)
        }
    } else if exp == 15 {
        // Inf or NaN
        if mant == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        // Normal: value = (1 + mant/8) * 2^(exp - 7)
        sign * (1.0 + mant / 8.0) * (2.0_f32).powi(exp - 7)
    }
}

/// 编译时数值模拟的顶层函数 (REQ-LC-011).
///
/// 对 DecodeTraceBuilder 生成的 TraceOp 序列进行标量模拟，
/// 验证量化解码的正确性。
///
/// # 参数
/// - `trace`: DecodeTraceBuilder 构建的 TraceOp 序列
/// - `desc`: 量化格式描述符
/// - `block_data`: 模拟 block 的字节数据 (seq_len=4, hidden=8 小规模)
/// - `inputs`: 输入值 [block_base_ptr, data_ptr, lane_offset?, high_bits_ptr?]
///
/// # 返回
/// 模拟结果，包含输出向量和 NaN/Inf 检测。
///
/// # 浮点精度
/// 容差 ±1e-5 (对比标量参考实现时使用)
pub fn simulate_compile(
    trace: &[TraceOp],
    desc: &QuantFormatDescriptor,
    block_data: &[u8],
    inputs: &[i64],
) -> Result<SimResult, CompilerError> {
    let sim = NumericalSimulator::new();
    sim.simulate_trace(trace, desc, block_data, inputs)
}

/// 编译时数值模拟器 (REQ-LC-011).
///
/// 对 TraceOp 序列进行标量模拟，验证量化解码的正确性。
///
/// # 使用方式
/// ```
/// let sim = NumericalSimulator::new();
/// let result = sim.simulate_trace(&trace, &desc, block_data, &inputs)?;
/// sim.verify_result(&result)?;
/// ```
pub struct NumericalSimulator {
    _phantom: std::marker::PhantomData<()>,
}

impl NumericalSimulator {
    /// 创建新的模拟器。
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// FP8 E4M3 byte → F32 scalar conversion for numerical simulation.
    fn fp8_e4m3_to_f32(byte: u8) -> f32 {
        let sign = (byte >> 7) & 1;
        let exp = (byte >> 3) & 0xF;
        let mant = byte & 0x7;
        if exp == 0 && mant == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        if exp == 0 {
            let v = (mant as f32) * 2f32.powi(-9);
            return if sign != 0 { -v } else { v };
        }
        if exp == 15 {
            let inf_bits = (sign as u32) << 31 | 0x7F800000;
            if mant == 0 {
                return f32::from_bits(inf_bits);
            }
            return f32::from_bits(inf_bits | 0x00400000);
        }
        let f32_exp = (exp as u32).wrapping_add(120);
        let bits = (sign as u32) << 31 | (f32_exp << 23) | ((mant as u32) << 20);
        f32::from_bits(bits)
    }

    /// FP8 E5M2 byte → F32 scalar conversion for numerical simulation.
    fn fp8_e5m2_to_f32(byte: u8) -> f32 {
        let sign = (byte >> 7) & 1;
        let exp = (byte >> 2) & 0x1F;
        let mant = byte & 0x3;
        if exp == 0 && mant == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        if exp == 0 {
            let v = (mant as f32) * 2f32.powi(-16);
            return if sign != 0 { -v } else { v };
        }
        if exp == 31 {
            let inf_bits = (sign as u32) << 31 | 0x7F800000;
            if mant == 0 {
                return f32::from_bits(inf_bits);
            }
            return f32::from_bits(inf_bits | 0x00400000);
        }
        let f32_exp = (exp as u32).wrapping_add(112);
        let bits = (sign as u32) << 31 | (f32_exp << 23) | ((mant as u32) << 21);
        f32::from_bits(bits)
    }

    /// 模拟 TraceOp 序列 (REQ-LC-011).
    ///
    /// # 参数
    /// - `trace`: TraceOp 序列
    /// - `desc`: 量化格式描述符
    /// - `block_data`: 模拟的 block 数据 (字节数组)
    /// - `inputs`: 输入值 [block_base_ptr, data_ptr, lane_offset?, high_bits_ptr?]
    ///
    /// # 返回
    /// 模拟结果，包含输出向量和错误信息。
    pub fn simulate_trace(
        &self,
        trace: &[TraceOp],
        desc: &QuantFormatDescriptor,
        block_data: &[u8],
        inputs: &[i64],
    ) -> Result<SimResult, CompilerError> {
        let mut state = SimState::new();

        // 初始化内存: 将 block_data 复制到模拟内存的固定位置
        let block_base_addr = 4096i64; // 固定地址
        for (i, &byte) in block_data.iter().enumerate() {
            state.store_u8(block_base_addr + i as i64, byte)?;
        }

        // 预扫描 trace 找到最大 Input 索引，推进 next_id 跳过 Input slot 空间。
        // Input(N) 使用 ValueId(N) 作为 slot，中间操作用 alloc_id() 分配。
        // next_id 必须从 max_input_idx+1 开始，防止与 Input ValueId 碰撞。
        let max_input_idx = trace.iter().map(|op| {
            if let TraceOp::Input(idx) = op { *idx as u32 + 1 } else { 0 }
        }).max().unwrap_or(0);
        state.next_id = max_input_idx;

        // 准备输入值映射: Input(idx) → SimValue。
        // Input(N) 的 N 是输入参数编号，不是 trace 位置。模拟器在执行时
        // 将 Input(N) 映射为 ValueId(trace_pos)，并把 input_values[N] 存入。
        let input_values: Vec<SimValue> = vec![
            SimValue::Integer(inputs.first().copied().unwrap_or(block_base_addr)),  // Input(0) = block_base
            SimValue::Integer(inputs.get(1).copied().unwrap_or(block_base_addr)),   // Input(1) = data_ptr
            SimValue::Integer(inputs.get(2).copied().unwrap_or(0)),                 // Input(2) = lane_offset
            SimValue::Integer(inputs.get(3).copied().unwrap_or(block_base_addr)),   // Input(3) = high_bits_ptr
        ];

        // 执行 TraceOp 序列
        let mut output_id = None;
        for (i, op) in trace.iter().enumerate() {
            let trace_pos = i as u32;
            let result = match op {
                TraceOp::Input(idx) => {
                    let id = ValueId(trace_pos);
                    let val = input_values.get(*idx as usize)
                        .copied()
                        .unwrap_or(SimValue::Integer(0));
                    state.set(id, val);
                    Ok(Some(id))
                }
                _ => self.exec_op_with_pos(op, trace_pos, &mut state, desc)
            };
            let result = result.map_err(|e| {
                CompilerError::CodegenViolation(format!(
                    "{} (at trace op #{}: {:?})",
                    match &e { CompilerError::CodegenViolation(s) => s.clone(), _ => format!("{:?}", e) },
                    i,
                    op
                ))
            })?;
            if let Some(id) = result {
                output_id = Some(id);
            }
        }

        // 提取输出值
        // 假设最后一个操作产生输出向量
        // 由于我们是标量模拟，需要根据输出 lane 数量扩展
        let output_slots = output_id.ok_or_else(|| CompilerError::CodegenViolation(
            "Simulation error: no output produced".to_string()
        ))?;

        // 简化: 假设输出是一个标量值，广播到向量
        let scalar_val = state.get(output_slots).as_float()?;
        let lanes = (desc.block_size + 7) / 8; // 粗略估计 lane 数
        let outputs = vec![scalar_val; lanes.max(1)];

        // 检查 NaN/Inf
        let has_nan = outputs.iter().any(|&v| v.is_nan());
        let has_inf = outputs.iter().any(|&v| v.is_infinite());

        Ok(SimResult {
            outputs,
            has_nan,
            has_inf,
        })
    }

    /// 执行单个 TraceOp（自分配 ValueId）。用于测试和 Loop body 内部调用。
    fn exec_op(
        &self,
        op: &TraceOp,
        state: &mut SimState,
        desc: &QuantFormatDescriptor,
    ) -> Result<Option<ValueId>, CompilerError> {
        let pos = state.next_id;
        state.next_id += 1;
        match op {
            TraceOp::Input(idx) => {
                // Input 引用外部已设置的值，但用 next_id 作为本 op 的 ValueId
                let src_val = state.get(ValueId(*idx));
                state.set(ValueId(pos), src_val);
                Ok(Some(ValueId(pos)))
            }
            _ => self.exec_op_with_pos(op, pos, state, desc)
        }
    }

    /// 执行单个 TraceOp，结果存储在 ValueId(trace_pos)。
    /// trace_pos 对应 trace 数组索引，与 push_op 分配的 ValueId 一致。
    fn exec_op_with_pos(
        &self,
        op: &TraceOp,
        trace_pos: u32,
        state: &mut SimState,
        desc: &QuantFormatDescriptor,
    ) -> Result<Option<ValueId>, CompilerError> {
        match op {
            // 输入
            TraceOp::Input(idx) => {
                let id = ValueId(*idx);
                // 输入已在 set_input 中设置
                Ok(Some(id))
            }

            // 常量
            TraceOp::Const(c) => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(*c as f32));
                Ok(Some(id))
            }

            // 算术运算
            TraceOp::Add(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = a_val + b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Sub(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = a_val - b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Mul(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = a_val * b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Div(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                if b_val == 0.0 {
                    return Err(CompilerError::CodegenViolation(
                        "Simulation error: division by zero".to_string()
                    ));
                }
                let result = a_val / b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Pow(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                if a_val < 0.0 {
                    return Err(CompilerError::CodegenViolation(
                        "Simulation error: pow with negative base".to_string()
                    ));
                }
                let result = a_val.powf(b_val);
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Fma(a, b, c) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let c_val = state.get(*c).as_float()?;
                let result = a_val * b_val + c_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // 一元运算
            TraceOp::Neg(a) => {
                let a_val = state.get(*a).as_float()?;
                let result = -a_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Abs(a) => {
                let a_val = state.get(*a).as_float()?;
                let result = a_val.abs();
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Sqrt(a) => {
                let a_val = state.get(*a).as_float()?;
                if a_val < 0.0 {
                    return Err(CompilerError::CodegenViolation(
                        format!("Simulation error: sqrt of negative value {}", a_val)
                    ));
                }
                let result = a_val.sqrt();
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Rsqrt(a) => {
                let a_val = state.get(*a).as_float()?;
                if a_val <= 0.0 {
                    return Err(CompilerError::CodegenViolation(
                        format!("Simulation error: rsqrt of non-positive value {}", a_val)
                    ));
                }
                let result = 1.0 / a_val.sqrt();
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Max(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = a_val.max(b_val);
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::Min(a, b) => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = a_val.min(b_val);
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // 量化操作
            TraceOp::QuantBitAnd { lhs, rhs } => {
                let lhs_val = state.get(*lhs).as_integer()?;
                let rhs_val = state.get(*rhs).as_integer()?;
                let result = lhs_val & rhs_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantBitOr { lhs, rhs } => {
                let lhs_val = state.get(*lhs).as_integer()?;
                let rhs_val = state.get(*rhs).as_integer()?;
                let result = lhs_val | rhs_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantBroadcast { src, lanes: _ } => {
                let src_val = state.get(*src).as_float()?;
                // 标量模拟: 广播就是复制值
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(src_val));
                Ok(Some(id))
            }

            TraceOp::QuantCastF16toF32 { src } => {
                let src_val = state.get(*src).as_integer()?;
                let addr = src_val;
                let f32_val = state.load_f16(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(f32_val));
                Ok(Some(id))
            }

            TraceOp::QuantCastI8toF32 { src } => {
                let src_val = state.get(*src).as_integer()?;
                let addr = src_val;
                let i8_val = state.load_u8(addr)? as i8;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(i8_val as f32));
                Ok(Some(id))
            }

            TraceOp::QuantCastFp8toF32 { src, format } => {
                let src_val = state.get(*src).as_integer()?;
                let addr = src_val;
                let byte = state.load_u8(addr)?;
                let f32_val = match format {
                    Fp8Format::E4M3 => Self::fp8_e4m3_to_f32(byte),
                    Fp8Format::E5M2 => Self::fp8_e5m2_to_f32(byte),
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(f32_val));
                Ok(Some(id))
            }

            TraceOp::QuantExtractBits { src, bit_offset, bit_width } => {
                let src_val = state.get(*src).as_integer()?;
                let mask = (1i64 << bit_width) - 1;
                let result = (src_val >> bit_offset) & mask;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantIntDivConst { src, divisor } => {
                let src_val = state.get(*src).as_integer()?;
                if *divisor == 0 {
                    return Err(CompilerError::CodegenViolation(
                        "Simulation error: division by zero".to_string()
                    ));
                }
                let result = src_val / divisor;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantPtrAddOffset { base, offset_bytes } => {
                let base_val = state.get(*base).as_integer()?;
                let addr = base_val + offset_bytes;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(addr));
                Ok(Some(id))
            }

            TraceOp::QuantPtrAddDynamic { base, index } => {
                let base_val = state.get(*base).as_integer()?;
                let idx_val = state.get(*index).as_integer()?;
                let addr = base_val + idx_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(addr));
                Ok(Some(id))
            }

            TraceOp::QuantScalarLoad { ptr, offset_bytes } => {
                let ptr_val = state.get(*ptr).as_integer()?;
                let addr = ptr_val + offset_bytes;
                let byte_val = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(byte_val as i64));
                Ok(Some(id))
            }

            TraceOp::QuantDequantFma { acc, a, b } => {
                let acc_val = state.get(*acc).as_float()?;
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = acc_val + a_val * b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            TraceOp::QuantIntMul { src, factor } => {
                let src_val = state.get(*src).as_integer()?;
                let result = src_val * factor;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantInterleave { lo, hi } => {
                // 标量模拟: 简单返回 lo 值
                let lo_val = state.get(*lo);
                let id = ValueId(trace_pos);
                state.set(id, lo_val);
                Ok(Some(id))
            }

            TraceOp::QuantConcatSeq { lo, hi } => {
                // 标量模拟: 顺序拼接在标量上下文中与 interleave 相同 (返回 lo 值)
                let lo_val = state.get(*lo);
                let id = ValueId(trace_pos);
                state.set(id, lo_val);
                Ok(Some(id))
            }

            TraceOp::QuantAndMask { src, mask } => {
                let src_val = state.get(*src).as_integer()?;
                let result = src_val & (*mask as i64);
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantLoadF16toF32 { ptr, offset_bytes } => {
                let ptr_val = state.get(*ptr).as_integer()?;
                let addr = ptr_val + offset_bytes;
                let f32_val = state.load_f16(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(f32_val));
                Ok(Some(id))
            }

            TraceOp::QuantLoadI8toF32 { ptr, offset_bytes } => {
                let ptr_val = state.get(*ptr).as_integer()?;
                let addr = ptr_val + offset_bytes;
                let i8_val = state.load_u8(addr)? as i8;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(i8_val as f32));
                Ok(Some(id))
            }

            TraceOp::QuantLoadBytesVec { ptr, offset_bytes, count: _, signed } => {
                let ptr_val = state.get(*ptr).as_integer()?;
                let addr = ptr_val + offset_bytes;
                let byte_val = state.load_u8(addr)?;
                let val = if *signed {
                    (byte_val as i8) as i64
                } else {
                    byte_val as i64
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(val));
                Ok(Some(id))
            }

            TraceOp::QuantShiftLeft { src, amount } => {
                let src_val = state.get(*src).as_integer()?;
                let result = src_val << amount;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            TraceOp::QuantShiftRight { src, amount } => {
                let src_val = state.get(*src).as_integer()?;
                let result = src_val >> amount;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            // ── Exp ──
            TraceOp::Exp(a) => {
                let a_val = state.get(*a).as_float()?;
                let result = a_val.exp();
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── Tanh ──
            TraceOp::Tanh(a) => {
                let a_val = state.get(*a).as_float()?;
                let result = a_val.tanh();
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── Recip (倒数) ──
            TraceOp::Recip(a) => {
                let a_val = state.get(*a).as_float()?;
                if a_val == 0.0 {
                    return Err(CompilerError::CodegenViolation(
                        "Simulation error: reciprocal of zero".to_string()
                    ));
                }
                let result = 1.0 / a_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── Log ──
            TraceOp::Log(a) => {
                let a_val = state.get(*a).as_float()?;
                if a_val <= 0.0 {
                    return Err(CompilerError::CodegenViolation(
                        format!("Simulation error: log of non-positive value {}", a_val)
                    ));
                }
                let result = a_val.ln();
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── Sigmoid ──
            TraceOp::Sigmoid(a) => {
                let a_val = state.get(*a).as_float()?;
                let result = 1.0 / (1.0 + (-a_val).exp());
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── ConditionalBranch ──
            TraceOp::ConditionalBranch(mask_val, true_val, false_val) => {
                let m = state.get(*mask_val).as_float()?;
                let t = state.get(*true_val).as_float()?;
                let f = state.get(*false_val).as_float()?;
                let result = if m != 0.0 { t } else { f };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── QuantFma (混合精度 FMA) ──
            TraceOp::QuantFma { acc, act, weight, act_dtype: _, weight_dtype: _ } => {
                let acc_val = state.get(*acc).as_float()?;
                let act_val = state.get(*act).as_float()?;
                let weight_val = state.get(*weight).as_float()?;
                let result = acc_val + act_val * weight_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── BlockScale ──
            TraceOp::BlockScale { data, scale, block_size: _ } => {
                let data_val = state.get(*data).as_float()?;
                let scale_val = state.get(*scale).as_float()?;
                let result = data_val * scale_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── Cast (类型转换, 标量模拟直接传递) ──
            TraceOp::Cast { src, from: _, to: _ } => {
                let val = state.get(*src);
                match val {
                    SimValue::Float(_) => {
                        let id = ValueId(trace_pos);
                        state.set(id, val);
                        Ok(Some(id))
                    }
                    SimValue::Integer(i) => {
                        let id = ValueId(trace_pos);
                        state.set(id, SimValue::Float(i as f32));
                        Ok(Some(id))
                    }
                    SimValue::Invalid => Err(CompilerError::CodegenViolation(
                        "Simulation error: Cast on invalid value".to_string()
                    )),
                }
            }

            // ── HReduce (标量模拟: 直接取源值作为标量结果) ──
            TraceOp::HReduce { src, op } => {
                let val = state.get(*src).as_float()?;
                let result = match op {
                    ReduceKind::Sum | ReduceKind::Prod | ReduceKind::Max
                    | ReduceKind::Min | ReduceKind::LogSum => val,
                    ReduceKind::Count => 1.0,
                    ReduceKind::ArgMax => 0.0,
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── Prefetch (标量模拟: NOP) ──
            TraceOp::Prefetch { level: _ } => {
                // Prefetch 是内存层级提示，标量模拟不需要执行
                // 返回一个虚值以保持 SSA 连续性
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── NonTemporalStore (标量模拟: NOP) ──
            TraceOp::NonTemporalStore => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── BitExtract ──
            TraceOp::BitExtract { src, offset, width } => {
                let src_val = state.get(*src).as_integer()?;
                let mask = (1i64 << width) - 1;
                let result = (src_val >> offset) & mask;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            // ── Permute (标量模拟: 传递源值) ──
            TraceOp::Permute { src, indices: _ } => {
                let val = state.get(*src);
                let id = ValueId(trace_pos);
                state.set(id, val);
                Ok(Some(id))
            }

            // ── Compare ──
            TraceOp::Compare { a, b, op } => {
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = match op {
                    CmpOp::Eq => if a_val == b_val { 1.0 } else { 0.0 },
                    CmpOp::Ne => if a_val != b_val { 1.0 } else { 0.0 },
                    CmpOp::Lt => if a_val < b_val { 1.0 } else { 0.0 },
                    CmpOp::Le => if a_val <= b_val { 1.0 } else { 0.0 },
                    CmpOp::Gt => if a_val > b_val { 1.0 } else { 0.0 },
                    CmpOp::Ge => if a_val >= b_val { 1.0 } else { 0.0 },
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── MaskedOp (标量模拟: mask 为 true 则执行 op) ──
            TraceOp::MaskedOp { op, mask } => {
                let mask_val = state.get(*mask).as_float()?;
                if mask_val != 0.0 {
                    self.exec_op_with_pos(op, trace_pos, state, desc)
                } else {
                    let id = ValueId(trace_pos);
                    state.set(id, SimValue::Float(0.0));
                    Ok(Some(id))
                }
            }

            // ── AtomicAdd (标量模拟: 对 value 求和) ──
            TraceOp::AtomicAdd { addr: _, val } => {
                let val_val = state.get(*val).as_float()?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(val_val));
                Ok(Some(id))
            }

            // ── FWHT (Fast Walsh-Hadamard Transform, 标量模拟: 传递源值) ──
            TraceOp::FWHT { src, dim: _ } => {
                let val = state.get(*src);
                let id = ValueId(trace_pos);
                state.set(id, val);
                Ok(Some(id))
            }

            // ── ScalarLoad ──
            TraceOp::ScalarLoad { base, offset } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                let addr = base_val + offset_val;
                let byte_val = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(byte_val as i64));
                Ok(Some(id))
            }

            // ── StrideMul ──
            TraceOp::StrideMul { value, stride } => {
                let value_val = state.get(*value).as_integer()?;
                let result = value_val * (*stride as i64);
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            // ── PtrAdd ──
            TraceOp::PtrAdd { base, offset } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                let addr = base_val + offset_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(addr));
                Ok(Some(id))
            }

            // ── VecLoadIndexed (标量模拟: 加载一个字节) ──
            TraceOp::VecLoadIndexed { base, offset } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                let addr = base_val + offset_val;
                let byte_val = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(byte_val as f32));
                Ok(Some(id))
            }

            // ── VecStoreIndexed (标量模拟: 存储一个字节) ──
            TraceOp::VecStoreIndexed { base, offset, value } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                let value_val = state.get(*value).as_float()?;
                let addr = base_val + offset_val;
                state.store_u8(addr, value_val as u8)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(value_val));
                Ok(Some(id))
            }

            // ── BroadcastScalar ──
            TraceOp::BroadcastScalar { src } => {
                let val = state.get(*src);
                let id = ValueId(trace_pos);
                state.set(id, val);
                Ok(Some(id))
            }

            // ── BroadcastLoad ──
            TraceOp::BroadcastLoad { base, offset } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                let addr = base_val + offset_val;
                let byte_val = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(byte_val as f32));
                Ok(Some(id))
            }

            // ── GatherLoad (标量模拟: 加载一个元素) ──
            TraceOp::GatherLoad { base, indices, stride } => {
                let base_val = state.get(*base).as_integer()?;
                let idx_val = state.get(*indices).as_integer()?;
                let addr = base_val + idx_val * (*stride as i64);
                let byte_val = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(byte_val as f32));
                Ok(Some(id))
            }

            // ── ScatterStore (标量模拟: 存储一个元素) ──
            TraceOp::ScatterStore { base, indices, value, stride } => {
                let base_val = state.get(*base).as_integer()?;
                let idx_val = state.get(*indices).as_integer()?;
                let value_val = state.get(*value).as_float()?;
                let addr = base_val + idx_val * (*stride as i64);
                state.store_u8(addr, value_val as u8)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(value_val));
                Ok(Some(id))
            }

            // ── TableLookup ──
            TraceOp::TableLookup { base, row_index, row_bytes } => {
                let base_val = state.get(*base).as_integer()?;
                let row_idx = state.get(*row_index).as_integer()?;
                let addr = base_val + row_idx * (*row_bytes as i64);
                let byte_val = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(byte_val as f32));
                Ok(Some(id))
            }

            // ── Mxfp4Dequant (标量模拟: 加载并反量化一个 MXFP4 元素) ──
            TraceOp::Mxfp4Dequant { data, scales, off_a: _, stride_a: _, off_b: _, stride_b: _, off_c: _, const_off: _, block_size: _ } => {
                let data_val = state.get(*data).as_integer()?;
                let scales_val = state.get(*scales).as_integer()?;
                // MXFP4: data 是 packed nibble 地址, scales 是 scale 地址
                let nibble = state.load_u8(data_val)?;
                let scale_byte = state.load_u8(scales_val)?;
                // E2M1 解码
                let e2m1_val = decode_e2m1_scalar(nibble & 0x0F);
                // E8M0 反量化: scale = 2^(scale_byte - 15)
                let scale = (2.0_f32).powi((scale_byte as i32) - 15);
                let result = e2m1_val * scale;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── BitAnd ──
            TraceOp::BitAnd(a, b) => {
                let a_val = state.get(*a).as_integer()?;
                let b_val = state.get(*b).as_integer()?;
                let result = a_val & b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(result));
                Ok(Some(id))
            }

            // ── QuantCodebookLookup ──
            TraceOp::QuantCodebookLookup { indices, codebook_data, vector_size: _, bits_per_entry: _ } => {
                let idx = state.get(*indices).as_integer()?;
                let codebook = *codebook_data;
                let codebook_idx = idx as usize;
                if codebook_idx < codebook.len() {
                    let val = codebook[codebook_idx] as f32;
                    let id = ValueId(trace_pos);
                    state.set(id, SimValue::Float(val));
                    Ok(Some(id))
                } else {
                    Err(CompilerError::CodegenViolation(
                        format!("Simulation error: codebook index {} out of bounds (len {})", codebook_idx, codebook.len())
                    ))
                }
            }

            // ── QuantE2m1LutDecode (修正实现) ──
            TraceOp::QuantE2m1LutDecode { packed_data_ptr, scale_byte, nvfp4_mode } => {
                let data_addr = state.get(*packed_data_ptr).as_integer()?;
                let scale_addr = state.get(*scale_byte).as_integer()?;
                let packed_byte = state.load_u8(data_addr)?;
                let scale_raw = state.load_u8(scale_addr)?;
                // E2M1 解码: nibble 包含 4-bit 值
                // 对于标量模拟，取低 4 位作为 E2M1 编码
                let nibble = packed_byte & 0x0F;
                let decoded = decode_e2m1_scalar(nibble);
                let scale = if *nvfp4_mode {
                    // NVFP4: UE4M3 scale (FP8 E4M3)
                    decode_fp8_e4m3(scale_raw)
                } else {
                    // MXFP4: E8M0 scale (无符号指数)
                    (2.0_f32).powi((scale_raw as i32) - 15)
                };
                let result = decoded * scale;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── QuantKQuantPackedScaleLookup (修正实现) ──
            TraceOp::QuantKQuantPackedScaleLookup { scales_base, sub_block_idx, scale_algo, selector } => {
                let base_addr = state.get(*scales_base).as_integer()?;
                let idx = state.get(*sub_block_idx).as_integer()?;
                let j = idx as usize;
                // K-Quant 6-bit packed scale 解码
                let result = match scale_algo {
                    PackedScaleAlgorithm::Q3KExtended => {
                        // Q3_K extended 格式: 不同的位布局
                        // 简化实现: 从 scales[j] 读取
                        let byte_val = state.load_u8(base_addr + j as i64)?;
                        byte_val as f32
                    }
                    PackedScaleAlgorithm::KQuant6Bit => match selector {
                        ScaleSelector::Min => {
                            // min 值: scales[j+4] & 0x3F (j<4)
                            //          (scales[j+4] >> 4) | ((scales[j] >> 6) << 4) (j>=4)
                            if j < 4 {
                                let byte_val = state.load_u8(base_addr + (j + 4) as i64)?;
                                (byte_val & 0x3F) as f32
                            } else {
                                let b_low = state.load_u8(base_addr + (j + 4) as i64)?;
                                let b_high = state.load_u8(base_addr + j as i64)?;
                                ((b_low >> 4) | ((b_high >> 6) << 4)) as f32
                            }
                        }
                        ScaleSelector::Scale => {
                            // scale 值: scales[j] & 0x3F (j<4)
                            //           (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4) (j>=4)
                            if j < 4 {
                                let byte_val = state.load_u8(base_addr + j as i64)?;
                                (byte_val & 0x3F) as f32
                            } else {
                                let b_low = state.load_u8(base_addr + (j + 4) as i64)?;
                                let b_high = state.load_u8(base_addr + (j - 4) as i64)?;
                                ((b_low & 0x0F) | ((b_high >> 6) << 4)) as f32
                            }
                        }
                    },
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── QuantScaleLoad ──
            TraceOp::QuantScaleLoad { source, offset, dtype: _ } => {
                let ptr = state.get(*source).as_integer()?;
                let addr = ptr + *offset as i64;
                let f32_val = state.load_f16(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(f32_val));
                Ok(Some(id))
            }

            // ── QuantDataLoad ──
            TraceOp::QuantDataLoad { source, offset, quant_type, block_size: _ } => {
                let ptr = state.get(*source).as_integer()?;
                let addr = ptr + *offset as i64;
                let desc = crate::quant_format::QuantFormatDescriptor::for_type(*quant_type);
                let byte = state.load_u8(addr)?;
                let val = match desc.data_kind {
                    QuantDataKind::Int8 => byte as i8 as f32,
                    QuantDataKind::PackedInt4 | QuantDataKind::SignedPackedInt4 => {
                        // Use low nibble as representative value for simulation
                        let nibble = byte & 0x0F;
                        if matches!(desc.data_kind, QuantDataKind::SignedPackedInt4) {
                            (nibble as i8 - 8) as f32
                        } else {
                            nibble as f32
                        }
                    }
                    _ => byte as f32,
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(val));
                Ok(Some(id))
            }

            // ── QuantZeroLoad ──
            TraceOp::QuantZeroLoad { source, offset, zp_type } => {
                let ptr = state.get(*source).as_integer()?;
                let addr = ptr + *offset as i64;
                let val = match zp_type {
                    ZeroLayout::None | ZeroLayout::StaticBias { .. } => 0.0f32,
                    ZeroLayout::BlockScalar { .. } | ZeroLayout::BlockMin { .. } | ZeroLayout::Hierarchical { .. } => {
                        state.load_f16(addr)?
                    }
                };
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(val));
                Ok(Some(id))
            }

            // ── QuantSubScaleLoad ──
            TraceOp::QuantSubScaleLoad { block_ptr, byte_offset, bits: _, sub_block_size: _ } => {
                let ptr = state.get(*block_ptr).as_integer()?;
                let addr = ptr + *byte_offset as i64;
                let val = state.load_f16(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(val));
                Ok(Some(id))
            }

            // ── QuantHighBitsLoad ──
            TraceOp::QuantHighBitsLoad { block_ptr, byte_offset, bits_per_elem: _ } => {
                let ptr = state.get(*block_ptr).as_integer()?;
                let addr = ptr + *byte_offset as i64;
                let byte = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Integer(byte as i64));
                Ok(Some(id))
            }

            // ── QuantCodebookDequant ──
            TraceOp::QuantCodebookDequant { indices, codebook_ptr, vector_size: _, bits_per_entry: _ } => {
                let idx = state.get(*indices).as_integer()?;
                let cb_ptr = state.get(*codebook_ptr).as_integer()?;
                let addr = cb_ptr + idx;
                let byte = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(byte as f32));
                Ok(Some(id))
            }

            // ── Loop (标量模拟: 展开一次迭代) ──
            TraceOp::Loop { bound: _, step_bytes: _, body } => {
                // 标量模拟: 仅执行 body 一次。
                // body 内的 op 共享外层 trace_pos 空间，使用 alloc_id 避免碰撞。
                let mut last_id = None;
                for inner_op in body {
                    let inner_pos = state.next_id;
                    state.next_id += 1;
                    match inner_op {
                        TraceOp::Input(idx) => {
                            // Loop body 内的 Input 引用外层输入
                            let val = state.get(ValueId(*idx));
                            let id = ValueId(inner_pos);
                            state.set(id, val);
                            last_id = Some(id);
                        }
                        _ => {
                            if let Ok(Some(id)) = self.exec_op_with_pos(inner_op, inner_pos, state, desc) {
                                last_id = Some(id);
                            }
                        }
                    }
                }
                Ok(last_id)
            }

            // ── PanelLoad (标量模拟: 加载 row 0, col 0 的一个元素) ──
            TraceOp::PanelLoad { base, offset, rows: _, cols: _ } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                let addr = base_val + offset_val;
                let byte = state.load_u8(addr)?;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(byte as f32));
                Ok(Some(id))
            }

            // ── PanelStore (标量模拟: 存储一个字节) ──
            TraceOp::PanelStore { base, offset, rows: _, cols: _ } => {
                let base_val = state.get(*base).as_integer()?;
                let offset_val = state.get(*offset).as_integer()?;
                // 标量模拟中，存储源的最后一个浮点值
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── PackBuffer (标量模拟: NOP) ──
            TraceOp::PackBuffer { src, dst: _, rows: _, cols: _, layout: _ } => {
                let val = state.get(*src);
                let id = ValueId(trace_pos);
                state.set(id, val);
                Ok(Some(id))
            }

            // ── SharedMemDeclare (标量模拟: NOP) ──
            TraceOp::SharedMemDeclare { name: _, bytes: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── AsyncCopyToShared (标量模拟: NOP) ──
            TraceOp::AsyncCopyToShared { name: _, src_offset: _, bytes: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── Tma2DCopy (标量模拟: NOP) ──
            TraceOp::Tma2DCopy { desc: _, coord_x: _, coord_y: _, bytes: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── AsyncWaitGroup (标量模拟: NOP) ──
            TraceOp::AsyncWaitGroup { n: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── SyncBarrier (标量模拟: NOP) ──
            TraceOp::SyncBarrier { name: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── TileConfig (标量模拟: NOP) ──
            TraceOp::TileConfig { rows: _, cols: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── TileMma (标量模拟: 标量 FMA; shape m/n/k 仅用于 VmInstr 透传, 标量模拟忽略) ──
            TraceOp::TileMma { c, a, b, m: _, n: _, k: _ } => {
                let c_val = state.get(*c).as_float()?;
                let a_val = state.get(*a).as_float()?;
                let b_val = state.get(*b).as_float()?;
                let result = c_val + a_val * b_val;
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(result));
                Ok(Some(id))
            }

            // ── TileRelease (标量模拟: NOP) ──
            TraceOp::TileRelease => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── Softmax (标量模拟: 标量恒等映射) ──
            TraceOp::Softmax { src, dst: _ } => {
                let s = state.get(*src).as_float()?;
                // 标量模拟: softmax(标量) = 1.0 (单个元素的 softmax 总是 1.0)
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(1.0));
                Ok(Some(id))
            }

            // ── EpilogueChain (标量模拟: 传递源值) ──
            TraceOp::EpilogueChain { ops: _ } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }

            // ── SPEC 24-QUANT-PIPELINE-JIT: QuantGather/QuantGemm structural ops ──
            // These are structural ops expanded by auto_select; numerical sim treats them as no-ops.
            TraceOp::QuantGather { .. } | TraceOp::QuantGemm { .. }
            | TraceOp::MtpDraft { .. }
            | TraceOp::MlaAttnScore { .. }
            | TraceOp::MlaRopeMerge { .. }
            | TraceOp::DynamicPrecisionSelect { .. }
            | TraceOp::QuantQ3KDecode { .. } => {
                let id = ValueId(trace_pos);
                state.set(id, SimValue::Float(0.0));
                Ok(Some(id))
            }
        }
    }

    /// 验证模拟结果 (REQ-LC-011).
    ///
    /// 检查输出是否包含 NaN/Inf，值域是否合理。
    pub fn verify_result(&self, result: &SimResult) -> Result<(), CompilerError> {
        if result.has_nan {
            return Err(CompilerError::CodegenViolation(
                "Numerical simulation failed: output contains NaN".to_string()
            ));
        }

        if result.has_inf {
            return Err(CompilerError::CodegenViolation(
                "Numerical simulation failed: output contains Inf".to_string()
            ));
        }

        // 检查所有值是否在合理范围内 (-1e6 到 1e6)
        for &val in &result.outputs {
            if val.abs() > 1e6 {
                return Err(CompilerError::CodegenViolation(
                    format!("Numerical simulation failed: output value {} out of reasonable range", val)
                ));
            }
        }

        Ok(())
    }
}

impl Default for NumericalSimulator {
    fn default() -> Self {
        Self::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §B VmInstr 数值解释器 (REQ-HW-TIER-006 / CR-TIER-SOVEREIGNTY-004)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// 与上方 NumericalSimulator (TraceOp 解释器, REQ-LC-011) 正交: 本节解释的是
// OpImpl::emit() 产出的 VmInstr 序列 (VmProgram), 而非 TraceOp。
//
// 用途: 跨硬件等级等价验证。每个 OpImpl 在给定 (FeatureSet, dtype) 下 emit 一段
// VmInstr 序列; 解释器以纯软件方式 (host 无关, 无需 AMX/AVX-512 硬件) 执行该序列,
// 产出数值结果, 与 scalar reference oracle 对齐。BF16 容差按 BF16 精度 (~1e-2) 定,
// 不要求 bit-exact (CR-TIER-SOVEREIGNTY-004)。
//
// 解释器覆盖的 VmInstr 子集 (GEMM-FMA 路径所需):
//   DeclareVReg / Broadcast / VecLoad / VecStore / Fma / VecNarrow / VecWiden /
//   Mov / LoopBegin / LoopEnd / TileConfig / TileMma / TileRelease
// 其余指令 (控制流/GPU/通信) 不在 GEMM-FMA 路径, 解释器遇之报错 (NO-SILENT-FALLBACK)。

use super::instr::{VRegId, SimdWidth, VmInstr, VmProgram, OffsetExpr, ScalarExpr, BoundExpr};
use crate::compiler::trace::QuantPrecision;

/// 解释器寄存器值: 一个 SIMD 向量用 Vec<f32> 表示 (lane 0..lanes-1)。
/// 标量寄存器 (SimdWidth::Scalar) 用单元素 Vec。
type RegValue = Vec<f32>;

/// tile vreg 的值 = 2D 矩阵块 (row-major f32 累加器)。
///
/// 解释器统一以 f32 累加 (AMX/Tensor Core 累加器均为 F32),
/// 输入精度 (BF16/F16) 通过 round-trip 窄化在 TileMma 内模拟。
/// `data[r * cols + c]` 为第 r 行第 c 列元素。
///
/// @trace REQ-HW-TIER-007 [req:VmInstr-TileLoad] tile 值模型, 解释器侧 2D 矩阵块表示
#[derive(Debug, Clone)]
struct TileVal {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl TileVal {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }
}

/// 解释器状态。
pub struct VmInterpState {
    /// VRegId → 寄存器值 (f32 lanes)。
    regs: HashMap<VRegId, RegValue>,
    /// 循环计数器 VRegId → 当前迭代索引 (0-based)。
    loop_counters: HashMap<VRegId, i64>,
    /// 循环字节偏移 VRegId → 当前字节偏移 (= counter × step_bytes)。
    loop_byte_offsets: HashMap<VRegId, i64>,
    /// 命名内存缓冲区: ptr_name ("a"/"b"/"c") → 字节 Vec。
    /// 解释器把 ABI 指针 VReg 映射到这些命名缓冲区, 而非裸地址, 避免与真实地址空间冲突。
    buffers: HashMap<String, Vec<u8>>,
    /// ptr VRegId → 缓冲区名 (解释器侧的指针解析)。
    ptr_names: HashMap<VRegId, String>,
    /// tile vreg VRegId → 2D 矩阵块值 (TileLoad 写入, TileMma 累加, TileStore 读取)。
    tile_regs: HashMap<VRegId, TileVal>,
}

impl VmInterpState {
    fn new() -> Self {
        Self {
            regs: HashMap::new(),
            loop_counters: HashMap::new(),
            loop_byte_offsets: HashMap::new(),
            buffers: HashMap::new(),
            ptr_names: HashMap::new(),
            tile_regs: HashMap::new(),
        }
    }

    /// 获取寄存器值 (克隆)。
    fn get_reg(&self, v: VRegId) -> Result<&[f32], CompilerError> {
        self.regs.get(&v).map(|v| v.as_slice()).ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "VmInterp: read of uninitialized VReg v{}", v.0
            ))
        })
    }

    /// 获取寄存器 lane 0 (标量)。
    fn get_scalar(&self, v: VRegId) -> Result<f32, CompilerError> {
        let lanes = self.get_reg(v)?;
        lanes.first().copied().ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "VmInterp: scalar read of empty VReg v{}", v.0
            ))
        })
    }

    /// 写入寄存器。
    fn set_reg(&mut self, v: VRegId, val: RegValue) {
        self.regs.insert(v, val);
    }

    /// 绑定 ptr VReg → 缓冲区名。
    fn bind_ptr(&mut self, v: VRegId, name: &str) {
        self.ptr_names.insert(v, name.to_string());
    }

    /// 解析 TileLoad/TileStore 的字节偏移 VReg (k_offset / out_offset)。
    ///
    /// 这些 VReg 是 emit_loop 分配的 byte_offset VReg (VRegKind::ByteOffset),
    /// 其值存于 loop_byte_offsets (= counter × step_bytes), 不在 regs 中。
    /// 优先取 loop_byte_offsets, 回退 loop_counters (取迭代索引, 兼容),
    /// 再回退 get_scalar (兼容静态偏移存于标量寄存器的场景)。
    fn resolve_byte_offset(&self, v: VRegId) -> Result<i64, CompilerError> {
        if let Some(bo) = self.loop_byte_offsets.get(&v) {
            return Ok(*bo);
        }
        if let Some(ct) = self.loop_counters.get(&v) {
            return Ok(*ct);
        }
        // 兼容: 偏移量以标量形式存于普通寄存器 (非循环绑定)。
        Ok(self.get_scalar(v)? as i64)
    }
}

/// 求值 OffsetExpr → 字节偏移 (i64)。
///
/// 语义对齐 x86_lower::eval_offset_to_rax: `OffsetExpr::LoopOffset(vreg)` 读取 vreg
/// 物理寄存器的当前值。在 emit_loop 语义下, counter VReg 持有迭代索引 (0-based),
/// byte_offset VReg 持有迭代索引 × step_bytes。GEMM emit 里两种 VReg 都可能出现在
/// LoopOffset 位置 (如 `Mul(LoopOffset(k_ctr), b_row_stride)` 取迭代索引再乘行步幅)。
///
/// 因此 LoopOffset 解析优先 byte_offsets, 回退 counters (兼容两种 emit 风格)。
fn eval_offset(off: &OffsetExpr, state: &VmInterpState) -> Result<i64, CompilerError> {
    match off {
        OffsetExpr::Const(n) => Ok(*n as i64),
        OffsetExpr::LoopOffset(v) => {
            // 优先 byte_offset (标准 emit_loop 风格), 回退 counter (取迭代索引)。
            if let Some(bo) = state.loop_byte_offsets.get(v) {
                Ok(*bo)
            } else if let Some(ct) = state.loop_counters.get(v) {
                Ok(*ct)
            } else {
                Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: LoopOffset v{} not in active loop scope", v.0
                )))
            }
        }
        OffsetExpr::Add(a, b) => Ok(eval_offset(a, state)? + eval_offset(b, state)?),
        OffsetExpr::Mul(inner, scale) => Ok(eval_offset(inner, state)? * *scale as i64),
        OffsetExpr::ScalarVReg(v) => state.loop_counters.get(v).copied().ok_or_else(|| {
            CompilerError::CodegenViolation(format!(
                "VmInterp: ScalarVReg v{} not a loop counter", v.0
            ))
        }),
    }
}

/// 求值 ScalarExpr → f32 (用于 Broadcast 源)。
///
/// `load_dtype`: 当 src=MemLoad 时, 按 load_dtype 解码字节 (广播时按当前元素的 dtype,
///               而非 buf 的原始存储 dtype; 解释器 buf 已按 encode_matrix_bytes 写入实际 dtype 字节)。
fn eval_scalar(se: &ScalarExpr, state: &VmInterpState, load_dtype: QuantPrecision) -> Result<f32, CompilerError> {
    match se {
        ScalarExpr::Const(c) => Ok(*c),
        ScalarExpr::MemLoad(base, off) => {
            let buf_name = state.ptr_names.get(base).ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: MemLoad base v{} not bound to a buffer", base.0
                ))
            })?;
            let byte_off = eval_offset(off, state)? as usize;
            load_elem_from_buffer(state, buf_name, byte_off, load_dtype)
        }
        ScalarExpr::ExtractLane0(v) => state.get_scalar(*v),
        ScalarExpr::VReg(v) => state.get_scalar(*v),
    }
}

/// 从命名缓冲区按 dtype 加载单个标量。
fn load_scalar_from_buffer(
    state: &VmInterpState,
    buf_name: &str,
    byte_off: usize,
) -> Result<f32, CompilerError> {
    let buf = state.buffers.get(buf_name).ok_or_else(|| {
        CompilerError::CodegenViolation(format!("VmInterp: buffer '{}' not set", buf_name))
    })?;
    // 默认按 F32 解码 (GEMM A/B 在解释器侧统一存 f32 字节; dtype 由 VecLoad.dtype 决定,
    // 但 Broadcast.MemLoad 取标量时按 buf 的实际 dtype)。这里用 4 字节 f32 little-endian。
    if byte_off + 4 > buf.len() {
        return Err(CompilerError::CodegenViolation(format!(
            "VmInterp: MemLoad OOB in buf '{}': off {} (len {})", buf_name, byte_off, buf.len()
        )));
    }
    let b = &buf[byte_off..byte_off + 4];
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// 按 dtype 元素宽度从缓冲区加载一个值并转 f32。
fn load_elem_from_buffer(
    state: &VmInterpState,
    buf_name: &str,
    byte_off: usize,
    dtype: QuantPrecision,
) -> Result<f32, CompilerError> {
    let buf = state.buffers.get(buf_name).ok_or_else(|| {
        CompilerError::CodegenViolation(format!("VmInterp: buffer '{}' not set", buf_name))
    })?;
    match dtype {
        QuantPrecision::F32 => {
            if byte_off + 4 > buf.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecLoad F32 OOB in '{}': off {} (len {})", buf_name, byte_off, buf.len()
                )));
            }
            let b = &buf[byte_off..byte_off + 4];
            Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        QuantPrecision::BF16 => {
            if byte_off + 2 > buf.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecLoad BF16 OOB in '{}': off {} (len {})", buf_name, byte_off, buf.len()
                )));
            }
            let bytes = [buf[byte_off], buf[byte_off + 1]];
            Ok(half::bf16::from_le_bytes(bytes).to_f32())
        }
        QuantPrecision::F16 => {
            if byte_off + 2 > buf.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecLoad F16 OOB in '{}': off {} (len {})", buf_name, byte_off, buf.len()
                )));
            }
            let bytes = [buf[byte_off], buf[byte_off + 1]];
            Ok(half::f16::from_le_bytes(bytes).to_f32())
        }
        _ => Err(CompilerError::CodegenViolation(format!(
            "VmInterp: VecLoad dtype {:?} not supported by interpreter", dtype
        ))),
    }
}

/// 按 dtype 元素宽度把 f32 值写入缓冲区 (VecStore)。
fn store_elem_to_buffer(
    state: &mut VmInterpState,
    buf_name: &str,
    byte_off: usize,
    val: f32,
    dtype: QuantPrecision,
) -> Result<(), CompilerError> {
    let buf = state.buffers.get_mut(buf_name).ok_or_else(|| {
        CompilerError::CodegenViolation(format!("VmInterp: buffer '{}' not set", buf_name))
    })?;
    match dtype {
        QuantPrecision::F32 => {
            if byte_off + 4 > buf.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecStore F32 OOB in '{}': off {} (len {})", buf_name, byte_off, buf.len()
                )));
            }
            let b = val.to_le_bytes();
            buf[byte_off..byte_off + 4].copy_from_slice(&b);
            Ok(())
        }
        QuantPrecision::BF16 => {
            if byte_off + 2 > buf.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecStore BF16 OOB in '{}': off {} (len {})", buf_name, byte_off, buf.len()
                )));
            }
            let bf = half::bf16::from_f32(val);
            let b = bf.to_le_bytes();
            buf[byte_off..byte_off + 2].copy_from_slice(&b);
            Ok(())
        }
        QuantPrecision::F16 => {
            if byte_off + 2 > buf.len() {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecStore F16 OOB in '{}': off {} (len {})", buf_name, byte_off, buf.len()
                )));
            }
            let f = half::f16::from_f32(val);
            let b = f.to_le_bytes();
            buf[byte_off..byte_off + 2].copy_from_slice(&b);
            Ok(())
        }
        _ => Err(CompilerError::CodegenViolation(format!(
            "VmInterp: VecStore dtype {:?} not supported by interpreter", dtype
        ))),
    }
}

/// 循环上界 → 迭代次数。仅支持 Const (GEMM emit 用 Const 上界)。
fn bound_to_iters(bound: &BoundExpr) -> Result<usize, CompilerError> {
    match bound {
        BoundExpr::Const(n) => Ok(*n),
        _ => Err(CompilerError::CodegenViolation(format!(
            "VmInterp: non-Const loop bound {:?} not supported (GEMM emit 用 Const 上界)", bound
        ))),
    }
}

/// 解释单条 VmInstr (在当前循环上下文内)。
///
/// `width` 参数保留用于需要全局宽度的兜底路径 (当前 GEMM-FMA 子集指令均自带 width 字段)。
fn exec_vm_instr(
    instr: &VmInstr,
    state: &mut VmInterpState,
    _width: SimdWidth,
) -> Result<(), CompilerError> {
    match instr {
        VmInstr::DeclareVReg { .. } => Ok(()), // 声明, 解释器无需动作

        VmInstr::Broadcast { dst, src, width: w, dtype, .. } => {
            let lanes = w.f32_lanes().max(1);
            let v = eval_scalar(src, state, *dtype)?;
            state.set_reg(*dst, vec![v; lanes]);
            Ok(())
        }

        VmInstr::VecLoad { dst, base, offset, width: w, dtype, predicate: None } => {
            let lanes = w.f32_lanes().max(1);
            let buf_name = state.ptr_names.get(base).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: VecLoad base v{} not bound", base.0
                ))
            })?;
            let base_off = eval_offset(offset, state)? as usize;
            let elem = dtype.elem_bytes();
            let mut out = Vec::with_capacity(lanes);
            for i in 0..lanes {
                let off = base_off + i * elem;
                out.push(load_elem_from_buffer(state, &buf_name, off, *dtype)?);
            }
            state.set_reg(*dst, out);
            Ok(())
        }

        VmInstr::VecLoad { predicate: Some(_), .. } => Err(CompilerError::CodegenViolation(
            "VmInterp: masked VecLoad not supported in GEMM-FMA cross-tier sim".to_string()
        )),

        VmInstr::VecStore { base, offset, src, width: w, dtype, predicate: None } => {
            let lanes = w.f32_lanes().max(1);
            let buf_name = state.ptr_names.get(base).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: VecStore base v{} not bound", base.0
                ))
            })?;
            let base_off = eval_offset(offset, state)? as usize;
            let vals = state.get_reg(*src)?.to_vec();
            if vals.len() < lanes {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: VecStore src v{} has {} lanes < {}", src.0, vals.len(), lanes
                )));
            }
            let elem = dtype.elem_bytes();
            for i in 0..lanes {
                store_elem_to_buffer(state, &buf_name, base_off + i * elem, vals[i], *dtype)?;
            }
            Ok(())
        }

        VmInstr::VecStore { predicate: Some(_), .. } => Err(CompilerError::CodegenViolation(
            "VmInterp: masked VecStore not supported in GEMM-FMA cross-tier sim".to_string()
        )),

        VmInstr::Fma { dst, acc, a, b, dtype: _ } => {
            // dst = acc + a × b (逐 lane)。GEMM 用 F32 累加 (acc_dtype), dtype 字段
            // 在 naive emit 里 = 输入 dtype (BF16/F32), 但 acc 寄存器始终是 F32 累加值。
            // 解释器按寄存器实际 f32 值计算, 无需 dtype 分支。
            let acc_v = state.get_reg(*acc)?.to_vec();
            let a_v = state.get_reg(*a)?.to_vec();
            let b_v = state.get_reg(*b)?.to_vec();
            let n = acc_v.len().min(a_v.len()).min(b_v.len());
            if n == 0 {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: Fma empty operand (acc v{} len {}, a v{} len {}, b v{} len {})",
                    acc.0, acc_v.len(), a.0, a_v.len(), b.0, b_v.len()
                )));
            }
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(acc_v[i] + a_v[i] * b_v[i]);
            }
            state.set_reg(*dst, out);
            Ok(())
        }

        VmInstr::VecNarrow { dst, src, dst_dtype, src_dtype: _, width: w, .. } => {
            // F32 累加器 → BF16/F16 存储: 截断到 dst_dtype 精度后, 重新转回 F32 表示
            // (解释器统一存 f32, 窄化效果用 round-trip 模拟)。
            let lanes = w.f32_lanes().max(1);
            let src_v = state.get_reg(*src)?.to_vec();
            let mut out = Vec::with_capacity(lanes);
            for i in 0..lanes.min(src_v.len()) {
                let narrowed = narrow_to_dtype(src_v[i], *dst_dtype);
                out.push(narrowed);
            }
            state.set_reg(*dst, out);
            Ok(())
        }

        VmInstr::VecWiden { dst, src, src_dtype, .. } => {
            // 窄 dtype → F32: 解释器 buf 已统一存 f32 字节, widen 在解释器侧是恒等
            // (BF16 字节加载时已 to_f32)。但 src 寄存器若已存了窄值, 这里取 lane 0 即可。
            let src_v = state.get_reg(*src)?.to_vec();
            // widen: src 的窄值 (BF16/F16 lanes 压在 F32 寄存器里) → F32。解释器直接传递。
            let _ = src_dtype; // 宽化是无损语义, 无需 round
            state.set_reg(*dst, src_v);
            Ok(())
        }

        VmInstr::Mov { dst, src, .. } => {
            let v = state.get_reg(*src)?.to_vec();
            state.set_reg(*dst, v);
            Ok(())
        }

        // ── Tile 路径 (AMX/SME/TC): 2D tile 值模型 + host FMA 累加 ──
        // TileConfig/TileRelease 在解释器侧是 NOP (tile 寄存器是抽象资源, 无数值语义)。
        // TileLoad/TileMma/TileStore 携带完整 shape, 解释器按 host FMA 模拟累加语义,
        // 输入按 dtype round-trip 窄化 (R4), 累加器始终 F32 (AMX/TC 累加器为 F32)。
        VmInstr::TileConfig { .. } | VmInstr::TileRelease => Ok(()),

        VmInstr::TileLoad { dst_tile, base_ptr, k_offset, row_stride, rows, cols, dtype } => {
            // 语义: dst_tile[r][c] = mem[base_ptr + k_offset + r*row_stride + c*elem_bytes]
            let buf_name = state.ptr_names.get(base_ptr).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: TileLoad base v{} not bound", base_ptr.0
                ))
            })?;
            let base_off = state.resolve_byte_offset(*k_offset)? as usize;
            let qp = dtype.to_quant_precision();
            let eb = dtype.size_bytes();
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..*rows {
                for c in 0..*cols {
                    let byte_off = base_off + r * row_stride + c * eb;
                    // 按 dtype 解码字节 → f32, 然后做 round-trip 窄化 (R4) 模拟硬件精度。
                    let v = load_elem_from_buffer(state, &buf_name, byte_off, qp)?;
                    let narrowed = narrow_to_dtype(v, qp);
                    data.push(narrowed);
                }
            }
            state.tile_regs.insert(*dst_tile, TileVal { rows: *rows, cols: *cols, data });
            Ok(())
        }

        VmInstr::TileMma { c, a, b, m, n, k, dtype } => {
            // 语义: c[m][n] += sum_k a[m][k] × b[k][n], F32 累加器, 输入按 dtype 窄化。
            let qp = dtype.to_quant_precision();
            // 取 a (m×k) 与 b (k×n)。先克隆以避免与 c 的可变借用冲突 (c 可能等于 a/b 的别名场景下需独立拷贝)。
            let ta = state.tile_regs.get(a).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: TileMma a v{} not loaded (use TileLoad first)", a.0
                ))
            })?;
            let tb = state.tile_regs.get(b).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: TileMma b v{} not loaded (use TileLoad first)", b.0
                ))
            })?;
            // 形状校验: a 为 m×k, b 为 k×n。
            if ta.rows != *m || ta.cols != *k {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: TileMma a shape {}x{} != {}x{}", ta.rows, ta.cols, m, k
                )));
            }
            if tb.rows != *k || tb.cols != *n {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: TileMma b shape {}x{} != {}x{}", tb.rows, tb.cols, k, n
                )));
            }
            // 取/初始化 c (m×n)。累加器 F32, 复用已有 tile 值 (跨 K 块累加), 不存在则 zeros。
            let tc = state
                .tile_regs
                .entry(*c)
                .or_insert_with(|| TileVal::zeros(*m, *n));
            if tc.rows != *m || tc.cols != *n {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: TileMma c shape {}x{} != {}x{}", tc.rows, tc.cols, m, n
                )));
            }
            for i in 0..*m {
                for j in 0..*n {
                    let mut acc = tc.data[i * n + j];
                    for kk in 0..*k {
                        let av = narrow_to_dtype(ta.data[i * k + kk], qp);
                        let bv = narrow_to_dtype(tb.data[kk * n + j], qp);
                        acc += av * bv; // F32 累加 (AMX/TC 累加器为 F32)
                    }
                    tc.data[i * n + j] = acc;
                }
            }
            Ok(())
        }

        VmInstr::TileStore { src_tile, base_ptr, out_offset, row_stride, rows, cols, dtype } => {
            // 语义: mem[base_ptr + out_offset + r*row_stride + c*elem_bytes] = src_tile[r][c]
            let tile = state.tile_regs.get(src_tile).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: TileStore src v{} not computed (use TileMma first)", src_tile.0
                ))
            })?;
            if tile.rows != *rows || tile.cols != *cols {
                return Err(CompilerError::CodegenViolation(format!(
                    "VmInterp: TileStore src shape {}x{} != {}x{}",
                    tile.rows, tile.cols, rows, cols
                )));
            }
            let buf_name = state.ptr_names.get(base_ptr).cloned().ok_or_else(|| {
                CompilerError::CodegenViolation(format!(
                    "VmInterp: TileStore base v{} not bound", base_ptr.0
                ))
            })?;
            let off = state.resolve_byte_offset(*out_offset)? as usize;
            let qp = dtype.to_quant_precision();
            let eb = dtype.size_bytes();
            for r in 0..*rows {
                for c in 0..*cols {
                    // 写回时按 dtype round-trip 窄化 (存储精度)。
                    let v = narrow_to_dtype(tile.data[r * cols + c], qp);
                    store_elem_to_buffer(state, &buf_name, off + r * row_stride + c * eb, v, qp)?;
                }
            }
            Ok(())
        }

        // ── 不在 GEMM-FMA 路径的指令: 报错 (NO-SILENT-FALLBACK) ──
        _ => Err(CompilerError::CodegenViolation(format!(
            "VmInterp: VmInstr {:?} not in GEMM-FMA cross-tier sim scope", instr
        ))),
    }
}

/// F32 → dst_dtype round-trip (模拟窄化的精度损失)。
fn narrow_to_dtype(val: f32, dst_dtype: QuantPrecision) -> f32 {
    match dst_dtype {
        QuantPrecision::BF16 => half::bf16::from_f32(val).to_f32(),
        QuantPrecision::F16 => half::f16::from_f32(val).to_f32(),
        _ => val, // F32/其他: 无损
    }
}

/// 解释 VmProgram: 执行指令序列, LoopBegin/LoopEnd 用迭代展开。
///
/// 返回最终状态 (供测试读取 C 缓冲区)。
// @trace REQ-HW-TIER-006 [req:VmInstr-Interpreter] VmInstr 数值解释器入口, 跨等级等价验证核心
pub fn interpret_vm_program(
    prog: &VmProgram,
    buffers: HashMap<String, Vec<u8>>,
    ptr_bindings: &[(VRegId, &str)], // (ptr_vreg, buffer_name)
    width: SimdWidth,
) -> Result<VmInterpState, CompilerError> {
    let mut state = VmInterpState::new();
    state.buffers = buffers;
    for (vreg, name) in ptr_bindings {
        state.bind_ptr(*vreg, name);
    }

    interpret_instr_slice(&prog.instrs, &mut state, width)?;
    Ok(state)
}

/// 解释指令切片 (支持嵌套循环)。
fn interpret_instr_slice(
    instrs: &[VmInstr],
    state: &mut VmInterpState,
    width: SimdWidth,
) -> Result<(), CompilerError> {
    let mut pc = 0;
    while pc < instrs.len() {
        match &instrs[pc] {
            VmInstr::LoopBegin { counter, byte_offset, bound, step_bytes } => {
                let iters = bound_to_iters(bound)?;
                let loop_end_idx = find_matching_loop_end(instrs, pc)
                    .ok_or_else(|| CompilerError::CodegenViolation(
                        format!("VmInterp: unmatched LoopBegin at pc {}", pc)
                    ))?;
                let body = &instrs[pc + 1..loop_end_idx];
                for i in 0..iters as i64 {
                    state.loop_counters.insert(*counter, i);
                    state.loop_byte_offsets.insert(*byte_offset, i * *step_bytes as i64);
                    interpret_instr_slice(body, state, width)?;
                }
                state.loop_counters.remove(counter);
                state.loop_byte_offsets.remove(byte_offset);
                pc = loop_end_idx + 1;
            }
            VmInstr::LoopEnd => {
                return Err(CompilerError::CodegenViolation(
                    format!("VmInterp: orphan LoopEnd at pc {}", pc)
                ));
            }
            other => {
                exec_vm_instr(other, state, width)?;
                pc += 1;
            }
        }
    }
    Ok(())
}

/// 从 pc 处的 LoopBegin 开始, 找到匹配的 LoopEnd 索引 (处理嵌套)。
fn find_matching_loop_end(instrs: &[VmInstr], loop_begin_pc: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = loop_begin_pc;
    while i < instrs.len() {
        match &instrs[i] {
            VmInstr::LoopBegin { .. } => depth += 1,
            VmInstr::LoopEnd => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §B.2 scalar GEMM reference oracle (CR-TIER-SOVEREIGNTY-004 标尺)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scalar GEMM reference: C[m][n] = Σ_p A[m][p] × B[p][n]。
///
/// 这是跨等级验证的**正确性 oracle** (标尺), 不是 native benchmark。所有 OpImpl
/// 等级 (GemmScalar / GemmFmaBlis / tile 路径) 的数值结果都对齐此 oracle (CR-004)。
///
/// 输入/输出按 row-major, F32 累加。`acc_dtype` 控制累加精度模拟:
/// - F32: 全程 f32 (F32 输入的精确标尺)
/// - BF16: 每次 mul-add 后 round 到 BF16 精度再继续累加 (模拟 BF16 WidenCompute 累加损失)
///         — 注: 真实 AVX2 BF16 路径用 F32 累加, 所以默认 acc=F32; 此参数仅供容差研究。
pub fn scalar_gemm_reference(
    a: &[f32], b: &[f32], m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for pi in 0..k {
                acc += a[mi * k + pi] * b[pi * n + ni];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

/// 把 f32 矩阵按 dtype 编码为字节缓冲区 (row-major)。
fn encode_matrix_bytes(data: &[f32], dtype: QuantPrecision) -> Vec<u8> {
    let elem = dtype.elem_bytes();
    let mut buf = vec![0u8; data.len() * elem];
    for (i, &v) in data.iter().enumerate() {
        let off = i * elem;
        match dtype {
            QuantPrecision::F32 => buf[off..off + 4].copy_from_slice(&v.to_le_bytes()),
            QuantPrecision::BF16 => {
                let bf = half::bf16::from_f32(v);
                buf[off..off + 2].copy_from_slice(&bf.to_le_bytes());
            }
            QuantPrecision::F16 => {
                let f = half::f16::from_f32(v);
                buf[off..off + 2].copy_from_slice(&f.to_le_bytes());
            }
            _ => {
                // 解释器仅支持 F32/BF16/F16 GEMM; 其他 dtype 在调用前应被过滤。
                buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    buf
}

/// 从字节缓冲区按 dtype 解码为 f32 矩阵 (row-major)。
fn decode_matrix_bytes(buf: &[u8], count: usize, dtype: QuantPrecision) -> Vec<f32> {
    let elem = dtype.elem_bytes();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * elem;
        match dtype {
            QuantPrecision::F32 => {
                out.push(f32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]));
            }
            QuantPrecision::BF16 => {
                out.push(half::bf16::from_le_bytes([buf[off], buf[off+1]]).to_f32());
            }
            QuantPrecision::F16 => {
                out.push(half::f16::from_le_bytes([buf[off], buf[off+1]]).to_f32());
            }
            _ => {
                out.push(f32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]));
            }
        }
    }
    out
}

/// 按 dtype 返回跨等级验证容差 (CR-TIER-SOVEREIGNTY-004: BF16 ~1e-2, F32 ~1e-5)。
pub fn tolerance_for(dtype: QuantPrecision) -> f32 {
    match dtype {
        QuantPrecision::F32 => 1e-5,
        QuantPrecision::BF16 => 1e-2,  // BF16: 7 mantissa bits → ~0.008 relative
        QuantPrecision::F16 => 1e-3,   // F16: 10 mantissa bits → ~0.001 relative
        _ => 1e-2,
    }
}

/// 跨等级等价验证: 解释 OpImpl emit 的 VmInstr 序列, 与 scalar oracle 对齐。
///
/// 流程:
/// 1. 构造 A/B 随机矩阵 → scalar_gemm_reference 得 golden C
/// 2. A/B 按 dtype 编码进 "a"/"b" 缓冲区, "c" 缓冲区清零
/// 3. 解释 VmProgram (ptr_bindings 指向 a/b/c)
/// 4. 解码 "c" 缓冲区 → 实际 C
/// 5. 比对实际 C vs golden C, max abs diff ≤ tolerance_for(dtype)
///
/// 返回 (max_abs_diff, passed)。
// @trace REQ-HW-TIER-006 [req:CrossTier-Equivalence] OpImpl 跨等级等价验证, BF16 容差 ~1e-2
pub fn verify_op_impl_aligns_scalar(
    prog: &VmProgram,
    a: &[f32], b: &[f32],
    m: usize, n: usize, k: usize,
    dtype: QuantPrecision,
    width: SimdWidth,
    ptr_bindings: &[(VRegId, &str)],
) -> Result<(f32, bool), CompilerError> {
    let golden = scalar_gemm_reference(a, b, m, n, k);
    let a_bytes = encode_matrix_bytes(a, dtype);
    let b_bytes = encode_matrix_bytes(b, dtype);
    let c_elem_bytes = dtype.elem_bytes();
    let c_bytes = vec![0u8; m * n * c_elem_bytes];

    let mut buffers = HashMap::new();
    buffers.insert("a".to_string(), a_bytes);
    buffers.insert("b".to_string(), b_bytes);
    buffers.insert("c".to_string(), c_bytes);

    let state = interpret_vm_program(prog, buffers, ptr_bindings, width)?;
    // 读取 C
    let c_buf = state.buffers.get("c").ok_or_else(|| {
        CompilerError::CodegenViolation("VmInterp: c buffer missing after run".to_string())
    })?;
    let actual = decode_matrix_bytes(c_buf, m * n, dtype);

    // 跨等级等价: actual 与 golden 都已按 dtype 存储解码 (BF16 存储的 C 读回就是 BF16
    // 精度的 f32)。golden 用 f32 精确累加, 但 actual 经 VecNarrow 写成 BF16。两者差异
    // = 输出 dtype 精度。为公平比较, golden 也 round 到 dtype (模拟输出存储精度)。
    let mut max_diff = 0.0f32;
    let mut max_abs = 1.0f32;
    for i in 0..m * n {
        let g_rounded = narrow_to_dtype(golden[i], dtype);
        let d = (actual[i] - g_rounded).abs();
        if d > max_diff {
            max_diff = d;
        }
        max_abs = max_abs.max(golden[i].abs());
    }
    // 相对容差 (CR-TIER-SOVEREIGNTY-004): tol = max(abs_tol, relative_eps × |value|)。
    // BF16: 7 mantissa bits → eps ≈ 2^-7 ≈ 0.0078; 取 2% 相对 (留累加噪声裕度)。
    // F32:  23 mantissa bits → eps ≈ 1.2e-7; 纯 f32 路径应近 bit-exact (≤1e-5)。
    let abs_tol = tolerance_for(dtype);
    let rel_eps = match dtype {
        QuantPrecision::F32 => 1e-5,
        QuantPrecision::BF16 => 2e-2,
        QuantPrecision::F16 => 2e-3,
        _ => 2e-2,
    };
    let tol = abs_tol.max(rel_eps * max_abs);
    Ok((max_diff, max_diff <= tol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::trace::QuantPrecision;
    use crate::quant_format::{
        DataLayout, QuantDataKind, QuantFormatDescriptor, ScaleLayout, ZeroLayout,
        StorageLayout,
    };
    use crate::quant::QuantType;

    /// 创建一个最小默认量化格式描述符（用于不需要特定格式的测试）。
    fn test_quant_desc() -> QuantFormatDescriptor {
        QuantFormatDescriptor {
            name: "test",
            quant_type: QuantType::F32,
            block_size: 32,
            block_bytes: 128,
            bits_per_element: 8,
            scale_layout: ScaleLayout::None,
            zero_layout: ZeroLayout::None,
            data_layout: DataLayout::Bytes { offset: 0, signed: true },
            data_kind: QuantDataKind::Float32,
            codebook: None,
            storage_layout: StorageLayout::RowMajor,
            native_isa: None,
        }
    }

    #[test]
    fn test_simulator_basic_arithmetic() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();

        // 设置输入
        state.set_input(0, SimValue::Float(2.0));
        state.set_input(1, SimValue::Float(3.0));

        // 构造简单的 trace: input0 + input1
        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Add(ValueId(0), ValueId(1)),
        ];

        let desc = test_quant_desc();
        // 执行
        let mut output_id = None;
        for op in &trace {
            if let Ok(Some(id)) = sim.exec_op(op, &mut state, &desc) {
                output_id = Some(id);
            }
        }

        assert!(output_id.is_some());
        let result = state.get(output_id.unwrap()).as_float().unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_detect_nan() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();

        // 设置输入: sqrt(-1) 会产生 NaN
        state.set_input(0, SimValue::Float(-1.0));

        let trace = vec![TraceOp::Sqrt(ValueId(0))];

        let desc = test_quant_desc();
        let result = sim.exec_op(&trace[0], &mut state, &desc);
        // sqrt(-1) 应该返回错误
        assert!(result.is_err());
    }

    #[test]
    fn test_simulator_division_by_zero() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();

        state.set_input(0, SimValue::Float(1.0));
        state.set_input(1, SimValue::Float(0.0));

        let trace = vec![TraceOp::Div(ValueId(0), ValueId(1))];

        let desc = test_quant_desc();
        let result = sim.exec_op(&trace[0], &mut state, &desc);
        // 除以零应该返回错误
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_result_passes() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![1.0, 2.0, 3.0],
            has_nan: false,
            has_inf: false,
        };

        assert!(sim.verify_result(&result).is_ok());
    }

    #[test]
    fn test_verify_result_fails_on_nan() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![1.0, f32::NAN, 3.0],
            has_nan: true,
            has_inf: false,
        };

        assert!(sim.verify_result(&result).is_err());
    }

    #[test]
    fn test_verify_result_fails_on_inf() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![1.0, f32::INFINITY, 3.0],
            has_nan: false,
            has_inf: true,
        };

        assert!(sim.verify_result(&result).is_err());
    }

    #[test]
    fn test_verify_result_fails_on_out_of_range() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![1.0, 2e6, 3.0],
            has_nan: false,
            has_inf: false,
        };

        assert!(sim.verify_result(&result).is_err());
    }

    #[test]
    fn test_simulate_compile_basic() {
        // 测试 simulate_compile 顶层函数
        // 构造一个简单的 trace: input0 + const(3.0)
        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Const(3.0),
            TraceOp::Add(ValueId(0), ValueId(1)),
            TraceOp::Mul(ValueId(3), ValueId(2)),
        ];

        let desc = test_quant_desc();
        let block_data = vec![0u8; 64];
        let inputs = vec![4096i64, 4096i64];

        let result = simulate_compile(&trace, &desc, &block_data, &inputs);
        assert!(result.is_ok());
        let sim_result = result.unwrap();
        assert!(!sim_result.has_nan);
        assert!(!sim_result.has_inf);
    }

    #[test]
    fn test_simulate_compile_nan_detection() {
        // 测试 NaN 检测: sqrt(-1) 应该产生 NaN
        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Sqrt(ValueId(0)),
        ];

        let desc = test_quant_desc();
        let block_data = vec![0u8; 64];
        // Input(-1) as float = -1.0, sqrt(-1) should error
        let inputs = vec![-1i64, 4096i64];

        let result = simulate_compile(&trace, &desc, &block_data, &inputs);
        // sqrt(-1) 应该返回错误
        assert!(result.is_err());
    }

    // ── NumericalSimulator constructor / Default trait ──

    #[test]
    fn test_simulator_new_creates_instance() {
        let _sim = NumericalSimulator::new();
    }

    #[test]
    fn test_simulator_default_trait() {
        let sim1 = NumericalSimulator::new();
        let sim2 = NumericalSimulator::default();
        // Both constructors should produce a valid simulator that can verify results
        let result = SimResult {
            outputs: vec![1.0],
            has_nan: false,
            has_inf: false,
        };
        assert!(sim1.verify_result(&result).is_ok());
        assert!(sim2.verify_result(&result).is_ok());
    }

    // ── SimResult construction and field access ──

    #[test]
    fn test_sim_result_fields_access() {
        let result = SimResult {
            outputs: vec![1.5, -2.3, 0.0],
            has_nan: false,
            has_inf: false,
        };
        assert_eq!(result.outputs.len(), 3);
        assert!((result.outputs[0] - 1.5).abs() < 1e-6);
        assert!((result.outputs[1] - (-2.3)).abs() < 1e-5);
        assert!(!result.has_nan);
        assert!(!result.has_inf);
    }

    #[test]
    fn test_sim_result_clone_preserves_fields() {
        let original = SimResult {
            outputs: vec![42.0, 7.5],
            has_nan: true,
            has_inf: true,
        };
        let cloned = original.clone();
        assert_eq!(cloned.outputs, original.outputs);
        assert_eq!(cloned.has_nan, original.has_nan);
        assert_eq!(cloned.has_inf, original.has_inf);
    }

    #[test]
    fn test_sim_result_debug_format() {
        let result = SimResult {
            outputs: vec![1.0],
            has_nan: false,
            has_inf: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("outputs"));
        assert!(debug_str.contains("has_nan"));
        assert!(debug_str.contains("has_inf"));
    }

    // ── verify_result: boundary values ──

    #[test]
    fn test_verify_result_passes_at_boundary_max() {
        let sim = NumericalSimulator::new();
        // Exactly at the 1e6 boundary should still pass (not strictly greater)
        let result = SimResult {
            outputs: vec![1e6],
            has_nan: false,
            has_inf: false,
        };
        assert!(sim.verify_result(&result).is_ok());
    }

    #[test]
    fn test_verify_result_passes_negative_boundary() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![-1e6],
            has_nan: false,
            has_inf: false,
        };
        assert!(sim.verify_result(&result).is_ok());
    }

    #[test]
    fn test_verify_result_fails_negative_out_of_range() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![-2e6],
            has_nan: false,
            has_inf: false,
        };
        assert!(sim.verify_result(&result).is_err());
    }

    #[test]
    fn test_verify_result_passes_empty_outputs() {
        let sim = NumericalSimulator::new();
        let result = SimResult {
            outputs: vec![],
            has_nan: false,
            has_inf: false,
        };
        assert!(sim.verify_result(&result).is_ok());
    }

    // ── Arithmetic operations via exec_op ──

    #[test]
    fn test_simulator_subtraction() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(10.0));
        state.set_input(1, SimValue::Float(3.0));

        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Sub(ValueId(0), ValueId(1)),
        ];
        let desc = test_quant_desc();

        let mut output_id = None;
        for op in &trace {
            if let Ok(Some(id)) = sim.exec_op(op, &mut state, &desc) {
                output_id = Some(id);
            }
        }

        let result = state.get(output_id.unwrap()).as_float().unwrap();
        assert!((result - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_multiplication() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(4.0));
        state.set_input(1, SimValue::Float(5.0));

        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Mul(ValueId(0), ValueId(1)),
        ];
        let desc = test_quant_desc();

        let mut output_id = None;
        for op in &trace {
            if let Ok(Some(id)) = sim.exec_op(op, &mut state, &desc) {
                output_id = Some(id);
            }
        }

        let result = state.get(output_id.unwrap()).as_float().unwrap();
        assert!((result - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_fma() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(2.0));
        state.set_input(1, SimValue::Float(3.0));
        state.set_input(2, SimValue::Float(10.0));

        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Input(2),
            TraceOp::Fma(ValueId(0), ValueId(1), ValueId(2)),
        ];
        let desc = test_quant_desc();

        let mut output_id = None;
        for op in &trace {
            if let Ok(Some(id)) = sim.exec_op(op, &mut state, &desc) {
                output_id = Some(id);
            }
        }

        // FMA: 2.0 * 3.0 + 10.0 = 16.0
        let result = state.get(output_id.unwrap()).as_float().unwrap();
        assert!((result - 16.0).abs() < 1e-6);
    }

    // ── Unary operations ──

    #[test]
    fn test_simulator_neg() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(5.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Neg(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_abs() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(-7.5));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Abs(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_sqrt_positive() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(9.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Sqrt(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_sqrt_negative_errors() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(-4.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Sqrt(ValueId(0)), &mut state, &desc);
        assert!(result.is_err());
    }

    #[test]
    fn test_simulator_rsqrt() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(4.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Rsqrt(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_rsqrt_zero_errors() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Rsqrt(ValueId(0)), &mut state, &desc);
        assert!(result.is_err());
    }

    // ── Min / Max ──

    #[test]
    fn test_simulator_max() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(3.0));
        state.set_input(1, SimValue::Float(7.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Max(ValueId(0), ValueId(1)), &mut state, &desc);
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_min() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(3.0));
        state.set_input(1, SimValue::Float(7.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Min(ValueId(0), ValueId(1)), &mut state, &desc);
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 3.0).abs() < 1e-6);
    }

    // ── Transcendental operations ──

    #[test]
    fn test_simulator_exp() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Exp(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulator_tanh_zero() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Tanh(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_simulator_recip() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(4.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Recip(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_recip_zero_errors() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Recip(ValueId(0)), &mut state, &desc);
        assert!(result.is_err());
    }

    #[test]
    fn test_simulator_log_positive() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(std::f32::consts::E));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Log(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simulator_log_zero_errors() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Log(ValueId(0)), &mut state, &desc);
        assert!(result.is_err());
    }

    #[test]
    fn test_simulator_sigmoid_zero() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Sigmoid(ValueId(0)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 0.5).abs() < 1e-5);
    }

    // ── Compare operations ──

    #[test]
    fn test_simulator_compare_eq() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(5.0));
        state.set_input(1, SimValue::Float(5.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: CmpOp::Eq },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 1.0).abs() < 1e-6, "Eq should return 1.0 for equal values");
    }

    #[test]
    fn test_simulator_compare_lt() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(3.0));
        state.set_input(1, SimValue::Float(5.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: CmpOp::Lt },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 1.0).abs() < 1e-6, "Lt should return 1.0 when a < b");
    }

    #[test]
    fn test_simulator_compare_gt_false() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(2.0));
        state.set_input(1, SimValue::Float(5.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: CmpOp::Gt },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!(val.abs() < 1e-6, "Gt should return 0.0 when a < b");
    }

    // ── ConditionalBranch ──

    #[test]
    fn test_simulator_conditional_branch_true() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(1.0));  // mask
        state.set_input(1, SimValue::Float(42.0)); // true_val
        state.set_input(2, SimValue::Float(99.0)); // false_val

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2)),
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 42.0).abs() < 1e-6, "Should select true_val when mask != 0");
    }

    #[test]
    fn test_simulator_conditional_branch_false() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));  // mask
        state.set_input(1, SimValue::Float(42.0)); // true_val
        state.set_input(2, SimValue::Float(99.0)); // false_val

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::ConditionalBranch(ValueId(0), ValueId(1), ValueId(2)),
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 99.0).abs() < 1e-6, "Should select false_val when mask == 0");
    }

    // ── HReduce (scalar pass-through) ──

    #[test]
    fn test_simulator_hreduce_sum() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(7.5));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Sum },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 7.5).abs() < 1e-6, "HReduce Sum should pass through scalar value");
    }

    #[test]
    fn test_simulator_hreduce_count() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(7.5));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::HReduce { src: ValueId(0), op: ReduceKind::Count },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 1.0).abs() < 1e-6, "HReduce Count should return 1.0");
    }

    #[test]
    fn test_simulator_hreduce_argmax() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(7.5));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::HReduce { src: ValueId(0), op: ReduceKind::ArgMax },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 0.0).abs() < 1e-6, "HReduce ArgMax should return 0.0");
    }

    // ── Bit operations (integer domain) ──

    #[test]
    fn test_simulator_quant_bit_and() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(0xFF));
        state.set_input(1, SimValue::Integer(0x0F));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantBitAnd { lhs: ValueId(0), rhs: ValueId(1) },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 0x0F);
    }

    #[test]
    fn test_simulator_quant_bit_or() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(0xF0));
        state.set_input(1, SimValue::Integer(0x0F));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantBitOr { lhs: ValueId(0), rhs: ValueId(1) },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 0xFF);
    }

    #[test]
    fn test_simulator_quant_extract_bits() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(0xAB));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantExtractBits { src: ValueId(0), bit_offset: 4, bit_width: 4 },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 0x0A, "Should extract upper nibble of 0xAB");
    }

    // ── Pointer arithmetic ──

    #[test]
    fn test_simulator_quant_ptr_add_offset() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(1000));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantPtrAddOffset { base: ValueId(0), offset_bytes: 42 },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 1042);
    }

    // ── Division: valid and error ──

    #[test]
    fn test_simulator_division_valid() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(10.0));
        state.set_input(1, SimValue::Float(4.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Div(ValueId(0), ValueId(1)), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 2.5).abs() < 1e-6);
    }

    // ── SimValue: integer-to-float conversion via as_float ──

    #[test]
    fn test_sim_value_integer_as_float() {
        let val = SimValue::Integer(42);
        let f = val.as_float().unwrap();
        assert!((f - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_sim_value_invalid_as_float_errors() {
        let val = SimValue::Invalid;
        assert!(val.as_float().is_err());
    }

    // ── SimValue: float-to-integer conversion ──

    #[test]
    fn test_sim_value_float_as_integer_exact() {
        let val = SimValue::Float(7.0);
        let i = val.as_integer().unwrap();
        assert_eq!(i, 7);
    }

    #[test]
    fn test_sim_value_float_as_integer_non_integer_errors() {
        let val = SimValue::Float(3.14);
        assert!(val.as_integer().is_err());
    }

    #[test]
    fn test_sim_value_invalid_as_integer_errors() {
        let val = SimValue::Invalid;
        assert!(val.as_integer().is_err());
    }

    // ── SimValue: is_valid ──

    #[test]
    fn test_sim_value_is_valid() {
        assert!(SimValue::Float(1.0).is_valid());
        assert!(SimValue::Integer(1).is_valid());
        assert!(!SimValue::Invalid.is_valid());
    }

    // ── SimState: get returns Invalid for unset ──

    #[test]
    fn test_sim_state_get_unset_returns_invalid() {
        let state = SimState::new();
        let val = state.get(ValueId(999));
        assert!(!val.is_valid());
    }

    // ── SimState: memory load out-of-bounds errors ──

    #[test]
    fn test_sim_state_load_u8_out_of_bounds_negative() {
        let state = SimState::new();
        let result = state.load_u8(-1);
        assert!(result.is_err());
    }

    #[test]
    fn test_sim_state_load_u8_out_of_bounds_beyond_end() {
        let state = SimState::new();
        let result = state.load_u8(65536);
        assert!(result.is_err());
    }

    // ── SimState: store and load round-trip ──

    #[test]
    fn test_sim_state_store_load_roundtrip() {
        let mut state = SimState::new();
        state.store_u8(100, 0xAB).unwrap();
        let loaded = state.load_u8(100).unwrap();
        assert_eq!(loaded, 0xAB);
    }

    // ── SimState: set and get value round-trip ──

    #[test]
    fn test_sim_state_set_get_roundtrip() {
        let mut state = SimState::new();
        state.set(ValueId(42), SimValue::Float(3.14));
        let val = state.get(ValueId(42));
        assert!((val.as_float().unwrap() - 3.14).abs() < 1e-5);
    }

    // ── Cast operation ──

    #[test]
    fn test_simulator_cast_float_passthrough() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(2.5));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::Cast { src: ValueId(0), from: QuantPrecision::F32, to: QuantPrecision::F16 },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_simulator_cast_integer_to_float() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(7));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::Cast { src: ValueId(0), from: QuantPrecision::INT8, to: QuantPrecision::F32 },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 7.0).abs() < 1e-6);
    }

    // ── Const operation ──

    #[test]
    fn test_simulator_const() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();

        let desc = test_quant_desc();
        let result = sim.exec_op(&TraceOp::Const(3.14), &mut state, &desc);
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 3.14).abs() < 1e-5);
    }

    // ── BlockScale operation ──

    #[test]
    fn test_simulator_block_scale() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(2.0));
        state.set_input(1, SimValue::Float(3.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::BlockScale { data: ValueId(0), scale: ValueId(1), block_size: 32 },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 6.0).abs() < 1e-6, "BlockScale should multiply data * scale");
    }

    // ── E2M1 decode function ──

    #[test]
    fn test_decode_e2m1_scalar_zero() {
        // nibble 0: exp=0, value=0
        assert_eq!(decode_e2m1_scalar(0x00), 0.0);
    }

    #[test]
    fn test_decode_e2m1_scalar_positive_one() {
        // nibble 0x02: sign=0, exp=1, mant=0 → (1+0) * 2^(1-1) = 1.0
        let val = decode_e2m1_scalar(0x02);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_decode_e2m1_scalar_negative_one() {
        // nibble 0x0A: sign=1 (bit 3), exp=1 (bits 2:1), mant=0 → -1.0
        let val = decode_e2m1_scalar(0x0A);
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    // ── FP8 E4M3 decode function ──

    #[test]
    fn test_decode_fp8_e4m3_zero() {
        assert_eq!(decode_fp8_e4m3(0x00), 0.0);
    }

    #[test]
    fn test_decode_fp8_e4m3_one() {
        // sign=0, exp=127-120=7 bias adjusted, mant=0
        // exp=7 (bits 3-6), mant=0 → (1+0/8) * 2^(7-7) = 1.0
        let byte = 0b0_0111_000; // sign=0, exp=7, mant=0
        let val = decode_fp8_e4m3(byte);
        assert!((val - 1.0).abs() < 1e-5);
    }

    // ── Chained arithmetic: (a + b) * c ──

    #[test]
    fn test_simulator_chained_add_mul() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(2.0));
        state.set_input(1, SimValue::Float(3.0));
        state.set_input(2, SimValue::Float(4.0));

        // trace: input0, input1, input2, add(v0, v1), mul(v3, v2)
        let trace = vec![
            TraceOp::Input(0),
            TraceOp::Input(1),
            TraceOp::Input(2),
            TraceOp::Add(ValueId(0), ValueId(1)),
            TraceOp::Mul(ValueId(3), ValueId(2)),
        ];
        let desc = test_quant_desc();

        let mut output_id = None;
        for op in &trace {
            if let Ok(Some(id)) = sim.exec_op(op, &mut state, &desc) {
                output_id = Some(id);
            }
        }

        // (2.0 + 3.0) * 4.0 = 20.0
        let result = state.get(output_id.unwrap()).as_float().unwrap();
        assert!((result - 20.0).abs() < 1e-6);
    }

    // ── QuantFma mixed-precision ──

    #[test]
    fn test_simulator_quant_fma() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(10.0)); // acc
        state.set_input(1, SimValue::Float(2.0));  // act
        state.set_input(2, SimValue::Float(3.0));  // weight

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantFma {
                acc: ValueId(0),
                act: ValueId(1),
                weight: ValueId(2),
                act_dtype: QuantPrecision::BF16,
                weight_dtype: QuantPrecision::FP8E4M3,
            },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        // 10.0 + 2.0 * 3.0 = 16.0
        assert!((val - 16.0).abs() < 1e-6);
    }

    // ── TileMma (scalar FMA) ──

    #[test]
    fn test_simulator_tile_mma() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(1.0)); // c
        state.set_input(1, SimValue::Float(2.0)); // a
        state.set_input(2, SimValue::Float(3.0)); // b

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::TileMma { c: ValueId(0), a: ValueId(1), b: ValueId(2), m: 1, n: 1, k: 1 },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        // 1.0 + 2.0 * 3.0 = 7.0
        assert!((val - 7.0).abs() < 1e-6);
    }

    // ── Softmax (scalar: always 1.0) ──

    #[test]
    fn test_simulator_softmax_scalar() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(5.0));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::Softmax { src: ValueId(0), dst: ValueId(0) },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 1.0).abs() < 1e-6, "Softmax of single scalar should be 1.0");
    }

    // ── SimState i16 load ──

    #[test]
    fn test_sim_state_load_i16() {
        let mut state = SimState::new();
        // Store little-endian i16 value -1000 (= 0xFC18)
        let val: i16 = -1000;
        let bytes = val.to_le_bytes();
        state.store_u8(200, bytes[0]).unwrap();
        state.store_u8(201, bytes[1]).unwrap();
        let loaded = state.load_i16(200).unwrap();
        assert_eq!(loaded, -1000);
    }

    // ── SimState store out-of-bounds errors ──

    #[test]
    fn test_sim_state_store_u8_out_of_bounds() {
        let mut state = SimState::new();
        let result = state.store_u8(65536, 0xFF);
        assert!(result.is_err());
    }

    // ── QuantIntDivConst: valid and divide-by-zero error ──

    #[test]
    fn test_simulator_quant_int_div_const_valid() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(100));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantIntDivConst { src: ValueId(0), divisor: 7 },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 14, "100 / 7 = 14 (integer division)");
    }

    #[test]
    fn test_simulator_quant_int_div_const_zero_errors() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(42));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantIntDivConst { src: ValueId(0), divisor: 0 },
            &mut state, &desc,
        );
        assert!(result.is_err(), "Division by zero in QuantIntDivConst should error");
    }

    // ── QuantIntMul: integer multiplication ──

    #[test]
    fn test_simulator_quant_int_mul() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(3));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantIntMul { src: ValueId(0), factor: 11 },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 33, "3 * 11 = 33");
    }

    // ── QuantPtrAddDynamic: dynamic pointer offset ──

    #[test]
    fn test_simulator_quant_ptr_add_dynamic() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(4096));
        state.set_input(1, SimValue::Integer(128));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantPtrAddDynamic { base: ValueId(0), index: ValueId(1) },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 4224, "4096 + 128 = 4224");
    }

    // ── QuantShiftLeft / QuantShiftRight: bitwise shifts ──

    #[test]
    fn test_simulator_quant_shift_left() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(1));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantShiftLeft { src: ValueId(0), amount: 4 },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 16, "1 << 4 = 16");
    }

    #[test]
    fn test_simulator_quant_shift_right() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(0xFF00));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantShiftRight { src: ValueId(0), amount: 8 },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 0xFF, "0xFF00 >> 8 = 0xFF");
    }

    // ── QuantAndMask: bitwise AND with constant mask ──

    #[test]
    fn test_simulator_quant_and_mask() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(0xAB));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantAndMask { src: ValueId(0), mask: 0x0F },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 0x0B, "0xAB & 0x0F = 0x0B");
    }

    // ── QuantDequantFma: dequantize fused multiply-add ──

    #[test]
    fn test_simulator_quant_dequant_fma() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(5.0));  // acc
        state.set_input(1, SimValue::Float(1.5));  // a
        state.set_input(2, SimValue::Float(2.0));  // b

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::QuantDequantFma { acc: ValueId(0), a: ValueId(1), b: ValueId(2) },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 8.0).abs() < 1e-6, "5.0 + 1.5 * 2.0 = 8.0");
    }

    // ── BitExtract: general-purpose bit field extraction ──

    #[test]
    fn test_simulator_bit_extract() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(0b11110000));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::BitExtract { src: ValueId(0), offset: 4, width: 4 },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 0b1111, "Extract bits [7:4] of 0b11110000 = 0b1111");
    }

    // ── StrideMul: integer stride multiplication ──

    #[test]
    fn test_simulator_stride_mul() {
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Integer(5));

        let desc = test_quant_desc();
        let result = sim.exec_op(
            &TraceOp::StrideMul { value: ValueId(0), stride: 4 },
            &mut state, &desc,
        );
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_integer().unwrap();
        assert_eq!(val, 20, "5 * 4 (stride) = 20");
    }

    // ── SimValue arithmetic with special floats ──

    #[test]
    fn test_sim_value_float_as_integer_negative_exact() {
        // Arrange: a negative integer-valued float
        let val = SimValue::Float(-42.0);
        // Act: convert to integer
        let i = val.as_integer().unwrap();
        // Assert: -42 should convert correctly
        assert_eq!(i, -42);
    }

    #[test]
    fn test_sim_value_float_as_integer_overflow_errors() {
        // Arrange: a float value that exceeds i32::MAX (not an exact i32 integer)
        let val = SimValue::Float(3.5e9);
        // Act & Assert: value is outside i32 range, should error
        assert!(val.as_integer().is_err());
    }

    #[test]
    fn test_sim_value_integer_as_float_large() {
        // Arrange: a large integer value
        let val = SimValue::Integer(100000);
        // Act: convert to float
        let f = val.as_float().unwrap();
        // Assert: 100000 as f32 should be exact (within f32 precision for this range)
        assert!((f - 100000.0).abs() < 1.0);
    }

    // ── SimState: i16 load out-of-bounds ──

    #[test]
    fn test_sim_state_load_i16_out_of_bounds_negative_addr() {
        // Arrange: empty SimState
        let state = SimState::new();
        // Act & Assert: negative address should error
        assert!(state.load_i16(-1).is_err());
    }

    #[test]
    fn test_sim_state_load_i16_out_of_bounds_at_boundary() {
        // Arrange: empty SimState (65536 bytes)
        let state = SimState::new();
        // Act & Assert: addr 65535 would need bytes [65535, 65536], but 65536 is out-of-bounds
        assert!(state.load_i16(65535).is_err());
    }

    // ── E2M1 decode edge cases ──

    #[test]
    fn test_decode_e2m1_scalar_positive_six() {
        // Arrange: nibble 0x07 = sign=0, exp=3, mant=1 → (1+0.5) * 2^(3-1) = 1.5 * 4 = 6.0
        let nibble = 0x07;
        // Act
        let val = decode_e2m1_scalar(nibble);
        // Assert
        assert!((val - 6.0).abs() < 1e-6, "E2M1 nibble 0x07 should decode to 6.0");
    }

    #[test]
    fn test_decode_e2m1_scalar_negative_three() {
        // Arrange: nibble 0x0D = sign=1 (bit3=1), exp=2 (bits2:1=10), mant=1 → -(1+0.5) * 2^(2-1) = -3.0
        let nibble = 0x0D;
        // Act
        let val = decode_e2m1_scalar(nibble);
        // Assert
        assert!((val - (-3.0)).abs() < 1e-6, "E2M1 nibble 0x0D should decode to -3.0");
    }

    // ── FP8 E4M3 decode: Inf and NaN ──

    #[test]
    fn test_decode_fp8_e4m3_positive_infinity() {
        // Arrange: byte 0b0_1111_000 = sign=0, exp=15, mant=0 → +Inf
        let byte = 0x78;
        // Act
        let val = decode_fp8_e4m3(byte);
        // Assert
        assert!(val.is_infinite() && val.is_sign_positive(), "FP8 E4M3 byte 0x78 should decode to +Inf");
    }

    #[test]
    fn test_decode_fp8_e4m3_nan() {
        // Arrange: byte 0b0_1111_001 = sign=0, exp=15, mant=1 → NaN
        let byte = 0x79;
        // Act
        let val = decode_fp8_e4m3(byte);
        // Assert
        assert!(val.is_nan(), "FP8 E4M3 byte 0x79 should decode to NaN");
    }

    // ── Compare operations with zero and negative boundary values ──

    #[test]
    fn test_simulator_compare_le_with_equal_negative() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(-5.0));
        state.set_input(1, SimValue::Float(-5.0));
        let desc = test_quant_desc();
        // Act: compare -5.0 <= -5.0
        let result = sim.exec_op(
            &TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: CmpOp::Le },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        // Assert: equal negative values should satisfy Le
        assert!((val - 1.0).abs() < 1e-6, "Le should return 1.0 when a == b (both negative)");
    }

    #[test]
    fn test_simulator_compare_ge_zero_vs_negative() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));
        state.set_input(1, SimValue::Float(-1.0));
        let desc = test_quant_desc();
        // Act: compare 0.0 >= -1.0
        let result = sim.exec_op(
            &TraceOp::Compare { a: ValueId(0), b: ValueId(1), op: CmpOp::Ge },
            &mut state, &desc,
        );
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        // Assert: 0.0 >= -1.0 should be true
        assert!((val - 1.0).abs() < 1e-6, "Ge should return 1.0 when 0.0 >= -1.0");
    }

    // ── Unary operations: abs and sqrt with zero ──

    #[test]
    fn test_simulator_abs_zero() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));
        let desc = test_quant_desc();
        // Act: abs(0.0)
        let result = sim.exec_op(&TraceOp::Abs(ValueId(0)), &mut state, &desc);
        // Assert
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!(val.abs() < 1e-6, "abs(0.0) should be 0.0");
    }

    #[test]
    fn test_simulator_sqrt_zero() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(0.0));
        let desc = test_quant_desc();
        // Act: sqrt(0.0)
        let result = sim.exec_op(&TraceOp::Sqrt(ValueId(0)), &mut state, &desc);
        // Assert: sqrt(0.0) = 0.0, should succeed (0.0 is not negative)
        assert!(result.is_ok());
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!(val.abs() < 1e-6, "sqrt(0.0) should be 0.0");
    }

    // ── NumericalSimulator FP8 E5M2 decode ──

    #[test]
    fn test_simulator_fp8_e5m2_to_f32_zero() {
        // Arrange: byte 0x00 = sign=0, exp=0, mant=0 → 0.0
        // Act
        let val = NumericalSimulator::fp8_e5m2_to_f32(0x00);
        // Assert
        assert!(val == 0.0, "FP8 E5M2 byte 0x00 should decode to 0.0");
    }

    #[test]
    fn test_simulator_fp8_e5m2_to_f32_positive_infinity() {
        // Arrange: byte 0b0_11111_00 = sign=0, exp=31, mant=0 → +Inf
        let byte = 0x7C;
        // Act
        let val = NumericalSimulator::fp8_e5m2_to_f32(byte);
        // Assert
        assert!(val.is_infinite() && val.is_sign_positive(), "FP8 E5M2 byte 0x7C should decode to +Inf");
    }

    #[test]
    fn test_simulator_fp8_e5m2_to_f32_nan() {
        // Arrange: byte 0b0_11111_01 = sign=0, exp=31, mant=1 → NaN
        let byte = 0x7D;
        // Act
        let val = NumericalSimulator::fp8_e5m2_to_f32(byte);
        // Assert
        assert!(val.is_nan(), "FP8 E5M2 byte 0x7D should decode to NaN");
    }

    // ── Neg operation with boundary values ──

    #[test]
    fn test_simulator_neg_negative() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(-3.5));
        let desc = test_quant_desc();
        // Act: neg(-3.5)
        let result = sim.exec_op(&TraceOp::Neg(ValueId(0)), &mut state, &desc);
        // Assert
        let id = result.unwrap().unwrap();
        let val = state.get(id).as_float().unwrap();
        assert!((val - 3.5).abs() < 1e-6, "neg(-3.5) should be 3.5");
    }

    // ── Log with negative input errors ──

    #[test]
    fn test_simulator_log_negative_errors() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(-10.0));
        let desc = test_quant_desc();
        // Act & Assert: log of negative should error
        let result = sim.exec_op(&TraceOp::Log(ValueId(0)), &mut state, &desc);
        assert!(result.is_err());
    }

    // ── Rsqrt with negative input errors ──

    #[test]
    fn test_simulator_rsqrt_negative_errors() {
        // Arrange
        let sim = NumericalSimulator::new();
        let mut state = SimState::new();
        state.set_input(0, SimValue::Float(-4.0));
        let desc = test_quant_desc();
        // Act & Assert: rsqrt of negative should error
        let result = sim.exec_op(&TraceOp::Rsqrt(ValueId(0)), &mut state, &desc);
        assert!(result.is_err());
    }
}
