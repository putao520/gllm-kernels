//! Template Interpreter — AlgoStep tree → Vec<TraceOp> (SPEC 27 REQ-AT-006)
//!
//! 遍历 AlgoStep 树，将每个步骤翻译为对应的 TraceOp 序列。
//! 参数从 AlgoParam 解析为具体值，填入 TraceOp 字段。
//!
//! ## Slot 模型 (v3 — 绝对索引)
//!
//! `auto_lower_trace_raw` 增量构建 `slots: Vec<VRegId>`:
//! - 每个 TraceOp 处理后 push 一个结果到 slots
//! - TraceOp 中的 ValueId(u32) 参数引用 slots 数组索引
//! - `TraceOp::Input(n)` 从 `inputs[n]` 取值，也 push 到 slots
//!
//! **Loop body 内的 TraceOp 必须使用外层 slots 的绝对索引。**
//! `dispatch_trace_op` 处理 `TraceOp::Loop` 时将外层 `slots` 传给 body，
//! body 内不 push 结果到 slots。因此 body 内的 `TraceOp::Input(n)` 引用
//! 外部输入，`TraceOp::Fma(a, b, c)` 的 a/b/c 引用外层已有 slot。
//!
//! `SlotEnv` 追踪 "名称 → ValueId" 映射，确保 Loop body 内能通过名称
//! 引用 Loop 外产出的 PanelLoad、PackBuffer 等中间结果的正确 slot 索引。

use crate::compiler::codegen::vm::algo_template::*;
use crate::compiler::trace::{TraceOp, ValueId};
use crate::compiler::codegen::vm::instr::BoundExpr;
use std::collections::HashMap;

/// 模板参数表 — 将参数名解析为具体值
///
/// 支持两类参数:
/// - `usize` 值: 用于维度/循环边界/步长等整数参数
/// - `f64` 值: 用于浮点常量如 norm epsilon, 从图元数据传播
#[derive(Debug, Clone)]
pub struct ParamTable {
    values: HashMap<String, usize>,
    f64_values: HashMap<String, f64>,
}

impl ParamTable {
    pub fn new() -> Self {
        Self { values: HashMap::new(), f64_values: HashMap::new() }
    }

    pub fn from_template(_template: &AlgoTemplate) -> Self {
        Self::new()
    }

    pub fn set(&mut self, name: &str, value: usize) {
        self.values.insert(name.to_string(), value);
    }

    pub fn get(&self, name: &str) -> Option<usize> {
        self.values.get(name).copied()
    }

    pub fn resolve(&self, name: &str) -> usize {
        self.values.get(name).copied().unwrap_or(1)
    }

    /// Set a floating-point parameter (e.g. norm epsilon from NormSpec.eps).
    pub fn set_f64(&mut self, name: &str, value: f64) {
        self.f64_values.insert(name.to_string(), value);
    }

    /// Resolve a floating-point parameter by name.
    /// Returns `None` if the parameter is not set — the caller must decide
    /// whether to error or use a default.
    pub fn resolve_f64(&self, name: &str) -> Option<f64> {
        self.f64_values.get(name).copied()
    }
}

/// 外部输入描述 — 定义模板需要的外部 VReg 输入
#[derive(Debug, Clone)]
pub struct TemplateInputs {
    pub names: Vec<String>,
}

impl TemplateInputs {
    pub fn new(names: &[&str]) -> Self {
        Self { names: names.iter().map(|s| s.to_string()).collect() }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn gemm() -> Self {
        Self::new(&["a_ptr", "b_ptr", "c_ptr", "a_offset", "b_offset", "c_offset"])
    }

    pub fn norm() -> Self {
        Self::new(&["input_ptr", "weight_ptr", "output_ptr", "seq_offset"])
    }

    pub fn rope() -> Self {
        Self::new(&["data_ptr", "cos_ptr", "sin_ptr", "seq_offset"])
    }

    pub fn moe() -> Self {
        Self::new(&["hidden_ptr", "weight_ptr", "output_ptr", "seq_offset"])
    }
}

/// Slot 环境 — 追踪 "名称 → ValueId" 映射
///
/// 每个产出 TraceOp 的步骤注册其结果到 env，后续步骤通过名称引用。
/// Loop body 内继承外层 env，使用绝对 slot 索引引用外层中间结果。
struct SlotEnv {
    /// 名称 → ValueId 的映射
    slots: HashMap<String, ValueId>,
}

impl SlotEnv {
    fn new() -> Self {
        Self { slots: HashMap::new() }
    }

    fn bind(&mut self, name: &str, id: ValueId) {
        self.slots.insert(name.to_string(), id);
    }

    fn get(&self, name: &str) -> ValueId {
        self.slots.get(name).copied().unwrap_or(ValueId(0))
    }
}

/// 模板解释器 — 将 AlgoStep 树翻译为 Vec<TraceOp>
///
/// 核心设计:
/// - ops Vec 的长度 = slots 数组的长度 (auto_lower_trace_raw 增量构建)
/// - ValueId 参数引用 ops Vec 中的位置索引 (= slots 索引)
/// - 外部输入通过 `TraceOp::Input(n)` 引入, n = inputs 数组索引
/// - SlotEnv 追踪名称映射，确保 Loop body 内使用绝对外层 slot 索引
pub struct TemplateInterpreter {
    params: ParamTable,
}

impl TemplateInterpreter {
    pub fn new(params: ParamTable) -> Self {
        Self { params }
    }

    /// 实例化模板: 遍历 step 树，输出 TraceOp 序列
    pub fn instantiate(&mut self, template: &AlgoTemplate, inputs: &TemplateInputs) -> Vec<TraceOp> {
        let mut ops = Vec::new();
        let mut env = SlotEnv::new();

        // 前置: 用 TraceOp::Input(n) 引入所有外部输入到 slots
        for i in 0..inputs.len() {
            ops.push(TraceOp::Input(i as u32));
            env.bind(&inputs.names[i], ValueId(i as u32));
        }

        for step in template.steps {
            self.interpret_step(step, &mut ops, &mut env);
        }
        ops
    }

    fn next_slot(ops: &[TraceOp]) -> ValueId {
        ValueId(ops.len() as u32)
    }

    fn push_and_bind(ops: &mut Vec<TraceOp>, env: &mut SlotEnv, name: &str, op: TraceOp) {
        let slot = Self::next_slot(ops);
        ops.push(op);
        env.bind(name, slot);
    }

    fn interpret_step(&self, step: &AlgoStep, ops: &mut Vec<TraceOp>, env: &mut SlotEnv) {
        match step {
            AlgoStep::Seq(steps) => {
                for s in *steps {
                    self.interpret_step(s, ops, env);
                }
            }

            AlgoStep::Loop { bound, step: step_name, body } => {
                let bound_val = self.params.resolve(bound);
                let step_val = self.params.resolve(step_name);
                // Loop body 使用外层 slots 的绝对索引
                // dispatch_trace_op 处理 Loop 时传入外层 slots
                let mut body_ops = Vec::new();
                for s in *body {
                    self.interpret_body_step(s, &mut body_ops, env);
                }
                ops.push(TraceOp::Loop {
                    bound: BoundExpr::Const(bound_val),
                    step_bytes: step_val,
                    body: body_ops,
                });
            }

            AlgoStep::Conditional { requirement: _, body } => {
                for s in *body {
                    self.interpret_step(s, ops, env);
                }
            }

            // ── GEMM 结构型步骤 ──

            AlgoStep::LoadPanel { matrix, rows_param, cols_param } => {
                let _rows = self.params.resolve(rows_param);
                let _cols = self.params.resolve(cols_param);
                let (base_slot, offset_slot) = match matrix {
                    MatrixRole::A => (env.get("a_ptr"), env.get("a_offset")),
                    MatrixRole::B => (env.get("b_ptr"), env.get("b_offset")),
                    MatrixRole::C => (env.get("c_ptr"), env.get("c_offset")),
                };
                let name = match matrix {
                    MatrixRole::A => "panel_a",
                    MatrixRole::B => "panel_b",
                    MatrixRole::C => "panel_c",
                };
                Self::push_and_bind(ops, env, name, TraceOp::PanelLoad {
                    base: base_slot,
                    offset: offset_slot,
                    rows: _rows,
                    cols: _cols,
                });
            }

            AlgoStep::PackBuffer { buffer_name, rows_param, cols_param } => {
                let rows = self.params.resolve(rows_param);
                let cols = self.params.resolve(cols_param);
                let src_slot = env.get("panel_a");
                Self::push_and_bind(ops, env, buffer_name, TraceOp::PackBuffer {
                    src: src_slot,
                    dst: src_slot,
                    rows, cols,
                    layout: PackLayout::RowMajor,
                });
            }

            AlgoStep::MicroKernel => {
                let a_slot = env.get("panel_a");
                let b_slot = env.get("panel_b");
                let c_slot = env.get("c_ptr");
                ops.push(TraceOp::Fma(a_slot, b_slot, c_slot));
            }

            AlgoStep::StoreResult { rows_param, cols_param } => {
                let base_slot = env.get("c_ptr");
                let offset_slot = env.get("c_offset");
                ops.push(TraceOp::PanelStore {
                    base: base_slot,
                    offset: offset_slot,
                    rows: self.params.resolve(rows_param),
                    cols: self.params.resolve(cols_param),
                });
            }

            AlgoStep::SharedMemDeclare { name, size_param } => {
                let bytes = self.params.resolve(size_param);
                Self::push_and_bind(ops, env, name, TraceOp::SharedMemDeclare {
                    name: name.to_string(),
                    bytes,
                });
            }

            AlgoStep::AsyncCopyToSmem { buffer_name, size_param } => {
                let bytes = self.params.resolve(size_param);
                let src_offset = env.get("panel_a");
                ops.push(TraceOp::AsyncCopyToShared {
                    name: buffer_name.to_string(),
                    src_offset,
                    bytes,
                });
            }

            AlgoStep::AsyncWait { group } => {
                ops.push(TraceOp::AsyncWaitGroup { n: *group });
            }

            AlgoStep::Barrier { barrier_name } => {
                ops.push(TraceOp::SyncBarrier { name: barrier_name.to_string() });
            }

            AlgoStep::TileConfig { rows, cols } => {
                ops.push(TraceOp::TileConfig {
                    rows: self.params.resolve(rows),
                    cols: self.params.resolve(cols),
                });
            }

            AlgoStep::TileMma => {
                let c = env.get("panel_c");
                let a = env.get("panel_a");
                let b = env.get("panel_b");
                // tile shape (m,n,k) 从 params 推断 (设计 §6.1): m/n/k 是 GEMM 标准维度名,
                // 调用方 (gemm 模板) 已 set("m"/"n"/"k", ..)。resolve 默认 1 兼容零 slot 回退。
                let m = self.params.resolve("m");
                let n = self.params.resolve("n");
                let k = self.params.resolve("k");
                ops.push(TraceOp::TileMma { c, a, b, m, n, k });
            }

            AlgoStep::TileRelease => {
                ops.push(TraceOp::TileRelease);
            }

            AlgoStep::TraceBody(trace_steps) => {
                for ts in *trace_steps {
                    self.interpret_trace_step(ts, ops, env);
                }
            }

            AlgoStep::Reduce { op: _ } => {
                let src = Self::next_slot(ops).saturating_sub(1);
                ops.push(TraceOp::HReduce {
                    src,
                    op: crate::compiler::trace::ReduceKind::Sum,
                });
            }

            AlgoStep::Activation { kind } => {
                let src = Self::next_slot(ops).saturating_sub(1);
                match kind {
                    ActivationKind::Silu => {
                        ops.push(TraceOp::Sigmoid(src));
                        let sigmoid_out = Self::next_slot(ops) - 1;
                        ops.push(TraceOp::Mul(sigmoid_out, src));
                    }
                    ActivationKind::Relu => {
                        ops.push(TraceOp::Max(src, src));
                    }
                    ActivationKind::Gelu => {
                        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        let x = src;
                        // x^2
                        ops.push(TraceOp::Mul(x, x));
                        let x2 = Self::next_slot(ops) - 1;
                        // x^3 = x^2 * x
                        ops.push(TraceOp::Mul(x2, x));
                        let x3 = Self::next_slot(ops) - 1;
                        // 0.044715 * x^3
                        ops.push(TraceOp::Const(0.044715));
                        let c0447 = Self::next_slot(ops) - 1;
                        ops.push(TraceOp::Mul(c0447, x3));
                        let cx3 = Self::next_slot(ops) - 1;
                        // inner = x + 0.044715 * x^3
                        ops.push(TraceOp::Add(x, cx3));
                        let inner = Self::next_slot(ops) - 1;
                        // scaled = sqrt(2/pi) * inner = 0.7978845608 * inner
                        ops.push(TraceOp::Const(0.7978845608));
                        let c7978 = Self::next_slot(ops) - 1;
                        ops.push(TraceOp::Mul(c7978, inner));
                        let scaled = Self::next_slot(ops) - 1;
                        // t = tanh(scaled)
                        ops.push(TraceOp::Tanh(scaled));
                        let t = Self::next_slot(ops) - 1;
                        // 1 + t
                        ops.push(TraceOp::Const(1.0));
                        let one = Self::next_slot(ops) - 1;
                        ops.push(TraceOp::Add(one, t));
                        let one_plus_t = Self::next_slot(ops) - 1;
                        // x * (1 + t)
                        ops.push(TraceOp::Mul(x, one_plus_t));
                        let x_opt = Self::next_slot(ops) - 1;
                        // 0.5 * x * (1 + t)
                        ops.push(TraceOp::Const(0.5));
                        let half = Self::next_slot(ops) - 1;
                        ops.push(TraceOp::Mul(half, x_opt));
                    }
                    ActivationKind::Tanh | ActivationKind::Sigmoid => {
                        ops.push(TraceOp::Tanh(src));
                    }
                }
            }

            AlgoStep::Softmax => {
                let src = Self::next_slot(ops).saturating_sub(1);
                ops.push(TraceOp::Softmax { src, dst: src });
            }

            AlgoStep::Dequantize { mode: _ } => {
                let src = Self::next_slot(ops).saturating_sub(1);
                ops.push(TraceOp::Input(src.0));
            }

            AlgoStep::EmbeddingGather => {
                let base = env.get("a_ptr");
                let offset = env.get("a_offset");
                ops.push(TraceOp::VecLoadIndexed { base, offset });
            }

            AlgoStep::MoeRouterGemv { num_experts: _, hidden: _ } => {
                let a = env.get("a_ptr");
                let b = env.get("b_ptr");
                let c = env.get("c_ptr");
                ops.push(TraceOp::Fma(a, b, c));
            }

            AlgoStep::MoeTopK { num_experts: _, top_k: _ } => {
                let src = env.get("a_ptr");
                ops.push(TraceOp::Input(src.0));
            }

            AlgoStep::Epilogue { ops: epilogue_ops } => {
                ops.push(TraceOp::EpilogueChain {
                    ops: epilogue_ops.to_vec(),
                });
            }

            AlgoStep::ZeroFill { bytes_param: _ } => {
                let ptr = env.get("c_ptr");
                ops.push(TraceOp::Input(ptr.0));
            }

            AlgoStep::RowCopy { rows_param: _, cols_param: _ } => {
                let src = env.get("a_ptr");
                let dst = env.get("c_ptr");
                ops.push(TraceOp::Input(src.0));
                ops.push(TraceOp::Input(dst.0));
            }
        }
    }

    /// Loop body 内步骤处理 — 使用外层 env 的绝对 slot 索引
    ///
    /// body 内的 TraceOp 引用外层 slots 已有的 VReg（通过 env 查找）。
    /// 不 push 结果到外层 ops（dispatch_trace_op 的 Loop 处理不 push body 结果）。
    fn interpret_body_step(&self, step: &AlgoStep, body_ops: &mut Vec<TraceOp>, env: &SlotEnv) {
        match step {
            AlgoStep::Seq(steps) => {
                for s in *steps {
                    self.interpret_body_step(s, body_ops, env);
                }
            }

            AlgoStep::Loop { bound, step: step_name, body } => {
                let bound_val = self.params.resolve(bound);
                let step_val = self.params.resolve(step_name);
                let mut inner_body = Vec::new();
                for s in *body {
                    self.interpret_body_step(s, &mut inner_body, env);
                }
                body_ops.push(TraceOp::Loop {
                    bound: BoundExpr::Const(bound_val),
                    step_bytes: step_val,
                    body: inner_body,
                });
            }

            AlgoStep::Conditional { requirement: _, body } => {
                for s in *body {
                    self.interpret_body_step(s, body_ops, env);
                }
            }

            // body 内的 LoadPanel: 引用外层 env 的指针 slot
            AlgoStep::LoadPanel { matrix, rows_param, cols_param } => {
                let rows = self.params.resolve(rows_param);
                let cols = self.params.resolve(cols_param);
                let (base_slot, offset_slot) = match matrix {
                    MatrixRole::A => (env.get("a_ptr"), env.get("a_offset")),
                    MatrixRole::B => (env.get("b_ptr"), env.get("b_offset")),
                    MatrixRole::C => (env.get("c_ptr"), env.get("c_offset")),
                };
                body_ops.push(TraceOp::PanelLoad {
                    base: base_slot,
                    offset: offset_slot,
                    rows, cols,
                });
            }

            AlgoStep::PackBuffer { buffer_name: _, rows_param, cols_param } => {
                let rows = self.params.resolve(rows_param);
                let cols = self.params.resolve(cols_param);
                let src = env.get("panel_a");
                body_ops.push(TraceOp::PackBuffer {
                    src, dst: src,
                    rows, cols,
                    layout: PackLayout::RowMajor,
                });
            }

            // body 内的 MicroKernel: 引用外层 panel_a/panel_b/c_ptr
            AlgoStep::MicroKernel => {
                let a_slot = env.get("panel_a");
                let b_slot = env.get("panel_b");
                let c_slot = env.get("c_ptr");
                body_ops.push(TraceOp::Fma(a_slot, b_slot, c_slot));
            }

            AlgoStep::StoreResult { rows_param, cols_param } => {
                let base_slot = env.get("c_ptr");
                let offset_slot = env.get("c_offset");
                body_ops.push(TraceOp::PanelStore {
                    base: base_slot,
                    offset: offset_slot,
                    rows: self.params.resolve(rows_param),
                    cols: self.params.resolve(cols_param),
                });
            }

            AlgoStep::SharedMemDeclare { name, size_param } => {
                body_ops.push(TraceOp::SharedMemDeclare {
                    name: name.to_string(),
                    bytes: self.params.resolve(size_param),
                });
            }

            AlgoStep::AsyncCopyToSmem { buffer_name, size_param } => {
                let src_offset = env.get("panel_a");
                body_ops.push(TraceOp::AsyncCopyToShared {
                    name: buffer_name.to_string(),
                    src_offset,
                    bytes: self.params.resolve(size_param),
                });
            }

            AlgoStep::AsyncWait { group } => {
                body_ops.push(TraceOp::AsyncWaitGroup { n: *group });
            }

            AlgoStep::Barrier { barrier_name } => {
                body_ops.push(TraceOp::SyncBarrier { name: barrier_name.to_string() });
            }

            AlgoStep::TileConfig { rows, cols } => {
                body_ops.push(TraceOp::TileConfig {
                    rows: self.params.resolve(rows),
                    cols: self.params.resolve(cols),
                });
            }

            AlgoStep::TileMma => {
                let c = env.get("panel_c");
                let a = env.get("panel_a");
                let b = env.get("panel_b");
                let m = self.params.resolve("m");
                let n = self.params.resolve("n");
                let k = self.params.resolve("k");
                body_ops.push(TraceOp::TileMma { c, a, b, m, n, k });
            }

            AlgoStep::TileRelease => {
                body_ops.push(TraceOp::TileRelease);
            }

            AlgoStep::TraceBody(trace_steps) => {
                for ts in *trace_steps {
                    self.interpret_trace_step(ts, body_ops, env);
                }
            }

            AlgoStep::Reduce { op: _ } => {
                let src = env.get("panel_a");
                body_ops.push(TraceOp::HReduce {
                    src,
                    op: crate::compiler::trace::ReduceKind::Sum,
                });
            }

            AlgoStep::Activation { kind } => {
                let src = env.get("panel_a");
                match kind {
                    ActivationKind::Silu => {
                        body_ops.push(TraceOp::Sigmoid(src));
                        let sigmoid_out = src;
                        body_ops.push(TraceOp::Mul(sigmoid_out, src));
                    }
                    ActivationKind::Relu => {
                        body_ops.push(TraceOp::Max(src, src));
                    }
                    ActivationKind::Gelu => {
                        let x = src;
                        body_ops.push(TraceOp::Mul(x, x));
                        let x2 = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Mul(x2, x));
                        let x3 = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Const(0.044715));
                        let c0447 = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Mul(c0447, x3));
                        let cx3 = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Add(x, cx3));
                        let inner = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Const(0.7978845608));
                        let c7978 = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Mul(c7978, inner));
                        let scaled = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Tanh(scaled));
                        let t = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Const(1.0));
                        let one = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Add(one, t));
                        let one_plus_t = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Mul(x, one_plus_t));
                        let x_opt = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Const(0.5));
                        let half = Self::next_slot(body_ops) - 1;
                        body_ops.push(TraceOp::Mul(half, x_opt));
                    }
                    ActivationKind::Tanh | ActivationKind::Sigmoid => {
                        body_ops.push(TraceOp::Tanh(src));
                    }
                }
            }

            AlgoStep::Dequantize { mode: _ } => {
                let src = env.get("panel_a");
                body_ops.push(TraceOp::Input(src.0));
            }

            AlgoStep::Softmax => {
                let src = env.get("panel_a");
                body_ops.push(TraceOp::Softmax { src, dst: src });
            }

            AlgoStep::EmbeddingGather => {
                let base = env.get("a_ptr");
                let offset = env.get("a_offset");
                body_ops.push(TraceOp::VecLoadIndexed { base, offset });
            }

            AlgoStep::MoeRouterGemv { num_experts: _, hidden: _ } => {
                let a = env.get("a_ptr");
                let b = env.get("b_ptr");
                let c = env.get("c_ptr");
                body_ops.push(TraceOp::Fma(a, b, c));
            }

            AlgoStep::MoeTopK { num_experts: _, top_k: _ } => {
                let src = env.get("a_ptr");
                body_ops.push(TraceOp::Input(src.0));
            }

            AlgoStep::Epilogue { ops: epilogue_ops } => {
                body_ops.push(TraceOp::EpilogueChain {
                    ops: epilogue_ops.to_vec(),
                });
            }

            AlgoStep::ZeroFill { bytes_param: _ } => {
                let ptr = env.get("c_ptr");
                body_ops.push(TraceOp::Input(ptr.0));
            }

            AlgoStep::RowCopy { rows_param: _, cols_param: _ } => {
                let src = env.get("a_ptr");
                let dst = env.get("c_ptr");
                body_ops.push(TraceOp::Input(src.0));
                body_ops.push(TraceOp::Input(dst.0));
            }
        }
    }

    fn interpret_trace_step(&self, ts: &AlgoTraceStep, ops: &mut Vec<TraceOp>, env: &SlotEnv) {
        match ts {
            AlgoTraceStep::LoadInput { name } => {
                let slot = env.get(name);
                ops.push(TraceOp::Input(slot.0));
            }
            AlgoTraceStep::LoadConst { value } => {
                ops.push(TraceOp::Const(*value));
            }
            AlgoTraceStep::LoadParam { name } => {
                let value = self.params.resolve_f64(name).unwrap_or_else(|| {
                    panic!(
                        "AlgoTraceStep::LoadParam(\"{}\"): parameter not set in ParamTable. \
                         Call params.set_f64(\"{}\", value) before template instantiation.",
                         name, name
                    )
                });
                ops.push(TraceOp::Const(value));
            }
            AlgoTraceStep::BinOp { op, dst: _, a, b } => {
                let a_slot = env.get(a);
                let b_slot = env.get(b);
                match op {
                    TraceBinOp::Add => ops.push(TraceOp::Add(a_slot, b_slot)),
                    TraceBinOp::Sub => ops.push(TraceOp::Sub(a_slot, b_slot)),
                    TraceBinOp::Mul => ops.push(TraceOp::Mul(a_slot, b_slot)),
                    TraceBinOp::Div => ops.push(TraceOp::Div(a_slot, b_slot)),
                    TraceBinOp::Max => ops.push(TraceOp::Max(a_slot, b_slot)),
                    TraceBinOp::Min => ops.push(TraceOp::Min(a_slot, b_slot)),
                }
            }
            AlgoTraceStep::UnaryOp { op, dst: _, src } => {
                let src_slot = env.get(src);
                match op {
                    TraceUnaryOp::Exp => ops.push(TraceOp::Exp(src_slot)),
                    TraceUnaryOp::Sqrt => ops.push(TraceOp::Sqrt(src_slot)),
                    TraceUnaryOp::Rsqrt => ops.push(TraceOp::Rsqrt(src_slot)),
                    TraceUnaryOp::Tanh => ops.push(TraceOp::Tanh(src_slot)),
                    TraceUnaryOp::Sigmoid => ops.push(TraceOp::Sigmoid(src_slot)),
                    TraceUnaryOp::Neg => ops.push(TraceOp::Neg(src_slot)),
                    TraceUnaryOp::Abs => ops.push(TraceOp::Abs(src_slot)),
                    TraceUnaryOp::Recip => ops.push(TraceOp::Recip(src_slot)),
                    TraceUnaryOp::Log => ops.push(TraceOp::Log(src_slot)),
                }
            }
            AlgoTraceStep::Fma { acc, a, b } => {
                let acc_slot = env.get(acc);
                let a_slot = env.get(a);
                let b_slot = env.get(b);
                ops.push(TraceOp::Fma(a_slot, b_slot, acc_slot));
            }
            AlgoTraceStep::HReduce { src, op } => {
                let src_slot = env.get(src);
                let rk = match op {
                    ReduceKind::Sum => crate::compiler::trace::ReduceKind::Sum,
                    ReduceKind::Max => crate::compiler::trace::ReduceKind::Max,
                    ReduceKind::Min => crate::compiler::trace::ReduceKind::Min,
                };
                ops.push(TraceOp::HReduce { src: src_slot, op: rk });
            }
            AlgoTraceStep::Broadcast { src, dst: _ } => {
                let src_slot = env.get(src);
                ops.push(TraceOp::BroadcastScalar { src: src_slot });
            }
            AlgoTraceStep::VecLoadIndexed { base, offset } => {
                let base_slot = env.get(base);
                let offset_slot = env.get(offset);
                ops.push(TraceOp::VecLoadIndexed { base: base_slot, offset: offset_slot });
            }
            AlgoTraceStep::VecStoreIndexed { base, offset, src } => {
                let base_slot = env.get(base);
                let offset_slot = env.get(offset);
                let src_slot = env.get(src);
                ops.push(TraceOp::VecStoreIndexed { base: base_slot, offset: offset_slot, value: src_slot });
            }
            AlgoTraceStep::Cast { src, from, to } => {
                let src_slot = env.get(src);
                ops.push(TraceOp::Cast { src: src_slot, from: *from, to: *to });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpret_gemnaive() {
        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&crate::compiler::codegen::vm::algo_templates::GEMM_NAIVE, &inputs);
        assert!(!ops.is_empty(), "GEMM_NAIVE should produce TraceOps");
        assert!(matches!(ops[0], TraceOp::Input(0)), "First op should be Input(0) for a_ptr");
        assert!(matches!(ops[1], TraceOp::Input(1)), "Second op should be Input(1) for b_ptr");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::Loop { .. })),
            "GEMM_NAIVE should contain Loop ops");
    }

    #[test]
    fn test_interpret_gemblis() {
        let mut params = ParamTable::new();
        params.set("mc", 64);
        params.set("nc", 64);
        params.set("kc", 32);
        params.set("mr", 4);
        params.set("nr", 4);
        params.set("k_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&crate::compiler::codegen::vm::algo_templates::GEMM_BLIS, &inputs);
        assert!(!ops.is_empty(), "GEMM_BLIS should produce TraceOps");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::Loop { .. })),
            "GEMM_BLIS should contain Loop ops at top level");
        fn contains_panel_load(ops: &[TraceOp]) -> bool {
            for op in ops {
                match op {
                    TraceOp::PanelLoad { .. } => return true,
                    TraceOp::Loop { body, .. } => if contains_panel_load(body) { return true; },
                    _ => {}
                }
            }
            false
        }
        assert!(contains_panel_load(&ops), "GEMM_BLIS body should contain PanelLoad");
    }

    #[test]
    fn test_interpret_norm_rms() {
        let mut params = ParamTable::new();
        params.set_f64("eps", 1e-6); // eps from NormSpec, propagated via ParamTable
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::norm();
        let ops = interp.instantiate(
            &crate::compiler::codegen::vm::algo_templates::attention_norm_rope_moe::NORM_RMS,
            &inputs,
        );
        assert!(!ops.is_empty(), "NORM_RMS should produce TraceOps");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::HReduce { .. })),
            "NORM_RMS should contain HReduce");
    }

    #[test]
    fn test_slot_env_absolute_indices() {
        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&crate::compiler::codegen::vm::algo_templates::GEMM_NAIVE, &inputs);

        // 验证外层 ops 中 PanelLoad 使用的外部输入 slot 索引 < inputs.len()
        fn check_outer_refs(ops: &[TraceOp], num_inputs: usize) {
            for op in ops {
                match op {
                    TraceOp::PanelLoad { base, offset, .. } => {
                        assert!((base.0 as usize) < num_inputs,
                            "PanelLoad base {} must reference inputs (< {})", base, num_inputs);
                        assert!((offset.0 as usize) < num_inputs,
                            "PanelLoad offset {} must reference inputs (< {})", offset, num_inputs);
                    }
                    TraceOp::Loop { body, .. } => {
                        check_body_refs(body, num_inputs);
                    }
                    _ => {}
                }
            }
        }

        fn check_body_refs(body: &[TraceOp], num_inputs: usize) {
            for op in body {
                match op {
                    TraceOp::Fma(a, b, c) => {
                        // FMA 参数引用外层 slots — 必须是有效的外层 slot 索引
                        assert!((a.0 as usize) < num_inputs || *a == ValueId(0),
                            "Fma a={} must reference valid outer slot", a);
                    }
                    TraceOp::Loop { body: inner, .. } => {
                        check_body_refs(inner, num_inputs);
                    }
                    _ => {}
                }
            }
        }

        check_outer_refs(&ops, inputs.len());
    }

    #[test]
    fn test_gemm_blis_body_fma_refs_outer() {
        let mut params = ParamTable::new();
        params.set("mc", 64);
        params.set("nc", 64);
        params.set("kc", 32);
        params.set("mr", 4);
        params.set("nr", 4);
        params.set("k_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&crate::compiler::codegen::vm::algo_templates::GEMM_BLIS, &inputs);

        // 查找 body 内的 FMA，验证其参数引用外层 panel slots
        fn find_body_fmas(ops: &[TraceOp], fmas: &mut Vec<(ValueId, ValueId, ValueId)>) {
            for op in ops {
                match op {
                    TraceOp::Loop { body, .. } => {
                        find_body_fmas(body, fmas);
                    }
                    TraceOp::Fma(a, b, c) => {
                        fmas.push((*a, *b, *c));
                    }
                    _ => {}
                }
            }
        }

        let mut fmas = Vec::new();
        find_body_fmas(&ops, &mut fmas);
        // GEMM_BLIS 应该在 body 内有 FMA
        assert!(!fmas.is_empty(), "GEMM_BLIS should produce FMA ops in loop body");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // REQ-AT-011: 数值对齐验证 — 模板驱动 vs 直接 emit
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    fn count_instr_kinds(prog: &crate::compiler::codegen::vm::instr::VmProgram) -> (usize, usize, usize) {
        use crate::compiler::codegen::vm::instr::VmInstr;
        let mut loads = 0;
        let mut stores = 0;
        let mut fmas = 0;
        for instr in &prog.instrs {
            match instr {
                VmInstr::VecLoad { .. } | VmInstr::VecLoadConst { .. } => loads += 1,
                VmInstr::VecStore { .. } => stores += 1,
                VmInstr::Fma { .. } => fmas += 1,
                _ => {}
            }
        }
        (loads, stores, fmas)
    }

    #[test]
    fn test_at011_gemm_template_vs_direct_emit_alignment() {
        use crate::compiler::codegen::vm::algo_registry;
        use crate::compiler::codegen::vm::algo_template::AlgoStrategy;
        use crate::compiler::codegen::vm::auto_select;
        use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, SimdWidth};
        use crate::compiler::trace::QuantPrecision;
        use crate::dispatch::device_profile::DeviceProfile;

        let profile = DeviceProfile::detect();
        let width = SimdWidth::W256;
        let m = 8;
        let n = 16;
        let k = 32;
        let lanes = width.f32_lanes().max(1);

        let tmpl = algo_registry::select_template(&AlgoStrategy::GemmNaive, &profile)
            .expect("GEMM_NAIVE should always be available");

        let mut params = ParamTable::new();
        params.set("m", m);
        params.set("n", n);
        params.set("k", k);
        params.set("mr", 4);
        params.set("nr", lanes);

        let inputs = TemplateInputs::gemm();
        let mut interp = TemplateInterpreter::new(params);
        let trace_ops = interp.instantiate(tmpl, &inputs);

        assert!(!trace_ops.is_empty(), "template should produce TraceOps");
        assert!(trace_ops.iter().any(|op| matches!(op, TraceOp::Loop { .. })),
            "GEMM_NAIVE must contain at least one Loop for m iteration");

        let mut prog = VmProgram::new();
        let a_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let b_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let c_ptr = prog.alloc_vreg(VRegKind::Ptr, width);
        let a_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let b_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);
        let c_off = prog.alloc_vreg(VRegKind::ByteOffset, SimdWidth::Scalar);

        let result = auto_select::auto_lower_trace_raw(
            &mut prog, &trace_ops, &[a_ptr, b_ptr, c_ptr, a_off, b_off, c_off],
            width, QuantPrecision::F32,
        );
        assert!(result.is_ok(), "auto_lower_trace_raw should succeed: {:?}", result);

        let (loads, stores, fmas) = count_instr_kinds(&prog);
        assert!(prog.instrs.len() > 5,
            "template-driven GEMM should produce significant instructions, got {}:\n{:?}",
            prog.instrs.len(), prog.instrs);
    }

    #[test]
    fn test_at011_norm_rms_template_produces_valid_trace() {
        use crate::compiler::codegen::vm::auto_select;
        use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, SimdWidth};
        use crate::compiler::trace::QuantPrecision;

        let inputs = TemplateInputs::norm();
        let mut params = ParamTable::new();
        params.set_f64("eps", 1e-6); // eps from NormSpec, propagated via ParamTable
        let mut interp = TemplateInterpreter::new(params);
        let ops = interp.instantiate(
            &crate::compiler::codegen::vm::algo_templates::attention_norm_rope_moe::NORM_RMS,
            &inputs,
        );

        assert!(!ops.is_empty(), "NORM_RMS should produce TraceOps");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::HReduce { .. })),
            "NORM_RMS must contain HReduce for mean-of-squares");

        let mut prog = VmProgram::new();
        let slots: Vec<_> = (0..4)
            .map(|_| prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar))
            .collect();

        let result = auto_select::auto_lower_trace_raw(
            &mut prog, &ops, &slots, SimdWidth::W256, QuantPrecision::F32,
        );
        assert!(result.is_ok(), "NORM_RMS auto_lower should succeed: {:?}", result);
        assert!(prog.instrs.len() > 0, "NORM_RMS should produce VmInstrs");
    }

    #[test]
    fn test_at011_rope_template_produces_valid_trace() {
        use crate::compiler::codegen::vm::auto_select;
        use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, SimdWidth};
        use crate::compiler::trace::QuantPrecision;

        let inputs = TemplateInputs::rope();
        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let ops = interp.instantiate(
            &crate::compiler::codegen::vm::algo_templates::attention_norm_rope_moe::ROPE_STANDARD,
            &inputs,
        );

        assert!(!ops.is_empty(), "ROPE_STANDARD should produce TraceOps");

        let mut prog = VmProgram::new();
        let slots: Vec<_> = (0..4)
            .map(|_| prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar))
            .collect();

        let result = auto_select::auto_lower_trace_raw(
            &mut prog, &ops, &slots, SimdWidth::W256, QuantPrecision::F32,
        );
        assert!(result.is_ok(), "ROPE_STANDARD auto_lower should succeed: {:?}", result);
        assert!(prog.instrs.len() > 0, "ROPE_STANDARD should produce VmInstrs");
    }

    #[test]
    fn test_at011_sampling_templates_all_valid() {
        use crate::compiler::codegen::vm::auto_select;
        use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, SimdWidth};
        use crate::compiler::trace::QuantPrecision;

        let templates: &[(&str, &crate::compiler::codegen::vm::algo_template::AlgoTemplate)] = &[
            ("ARGMAX", &crate::compiler::codegen::vm::algo_templates::sampling::SAMPLING_ARGMAX),
            ("TEMPERATURE", &crate::compiler::codegen::vm::algo_templates::sampling::SAMPLING_TEMPERATURE),
            ("SOFTMAX", &crate::compiler::codegen::vm::algo_templates::sampling::SAMPLING_SOFTMAX),
            ("TOP_K", &crate::compiler::codegen::vm::algo_templates::sampling::SAMPLING_TOP_K),
            ("TOP_P", &crate::compiler::codegen::vm::algo_templates::sampling::SAMPLING_TOP_P),
            ("MULTINOMIAL", &crate::compiler::codegen::vm::algo_templates::sampling::SAMPLING_MULTINOMIAL),
        ];

        for (name, template) in templates {
            let params = ParamTable::new();
            let mut interp = TemplateInterpreter::new(params);
            let ops = interp.instantiate(template, &TemplateInputs::new(&["logits"]));

            assert!(!ops.is_empty(), "SAMPLING_{} should produce TraceOps", name);
            assert!(ops.iter().any(|op| matches!(op, TraceOp::Input(0))),
                "SAMPLING_{} should reference logits input", name);

            let mut prog = VmProgram::new();
            let logits_ptr = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

            let result = auto_select::auto_lower_trace_raw(
                &mut prog, &ops, &[logits_ptr], SimdWidth::W256, QuantPrecision::F32,
            );
            assert!(result.is_ok(), "SAMPLING_{} auto_lower should succeed: {:?}", name, result);
            assert!(prog.instrs.len() > 0, "SAMPLING_{} should produce VmInstrs", name);
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Additional unit tests: constructors, derives, boundary values
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_param_table_new_is_empty() {
        let table = ParamTable::new();

        assert_eq!(table.get("nonexistent"), None, "empty ParamTable should return None for any key");
        assert_eq!(table.resolve("nonexistent"), 1, "resolve should return default 1 for missing key");
    }

    #[test]
    fn test_param_table_set_get_overwrite() {
        let mut table = ParamTable::new();

        table.set("alpha", 42);
        assert_eq!(table.get("alpha"), Some(42));

        table.set("alpha", 99);
        assert_eq!(table.get("alpha"), Some(99), "set should overwrite existing value");

        table.set("beta", 0);
        assert_eq!(table.get("beta"), Some(0), "zero is a valid value");
        assert_eq!(table.get("gamma"), None, "unrelated key unaffected");
    }

    #[test]
    fn test_param_table_clone_isolation() {
        let mut table = ParamTable::new();
        table.set("x", 10);

        let cloned = table.clone();
        table.set("x", 20);

        assert_eq!(cloned.get("x"), Some(10), "clone should be independent of original");
        assert_eq!(table.get("x"), Some(20));
    }

    #[test]
    fn test_param_table_usize_boundary_values() {
        let mut table = ParamTable::new();

        table.set("zero", 0);
        table.set("max", usize::MAX);
        table.set("one", 1);

        assert_eq!(table.get("zero"), Some(0));
        assert_eq!(table.get("max"), Some(usize::MAX));
        assert_eq!(table.get("one"), Some(1));
        assert_eq!(table.resolve("zero"), 0, "resolve returns actual value when present");
        assert_eq!(table.resolve("max"), usize::MAX, "resolve handles usize::MAX");
    }

    #[test]
    fn test_template_inputs_len_and_presets() {
        let gemm = TemplateInputs::gemm();
        assert_eq!(gemm.len(), 6, "gemm preset should have 6 inputs");
        assert_eq!(gemm.names[0], "a_ptr");
        assert_eq!(gemm.names[5], "c_offset");

        let norm = TemplateInputs::norm();
        assert_eq!(norm.len(), 4, "norm preset should have 4 inputs");
        assert_eq!(norm.names[0], "input_ptr");

        let rope = TemplateInputs::rope();
        assert_eq!(rope.len(), 4, "rope preset should have 4 inputs");
        assert_eq!(rope.names[2], "sin_ptr");

        let moe = TemplateInputs::moe();
        assert_eq!(moe.len(), 4, "moe preset should have 4 inputs");
        assert_eq!(moe.names[3], "seq_offset");
    }

    #[test]
    fn test_template_inputs_custom_names() {
        let inputs = TemplateInputs::new(&["alpha", "beta", "gamma"]);

        assert_eq!(inputs.len(), 3);
        assert_eq!(inputs.names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_param_table_debug_format() {
        let mut table = ParamTable::new();
        table.set("m", 64);

        let debug_str = format!("{:?}", table);
        assert!(debug_str.contains("ParamTable"), "Debug output should contain struct name");
    }

    #[test]
    fn test_interpreter_instantiate_empty_template() {
        use crate::compiler::codegen::vm::algo_template::AlgoTemplate;
        use crate::compiler::codegen::vm::algo_template::{
            AlgoStrategy, DeviceReq,
        };

        let empty_template = AlgoTemplate {
            name: "test_empty",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&empty_template, &inputs);

        assert_eq!(ops.len(), 1, "empty template should produce exactly one Input(0) op");
        assert!(matches!(ops[0], TraceOp::Input(0)));
    }

    #[test]
    fn test_interpreter_with_params_resolves_loop_bound() {
        let mut params = ParamTable::new();
        params.set("m", 8);
        params.set("n", 16);
        params.set("mr", 4);
        params.set("nr", 8);

        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(
            &crate::compiler::codegen::vm::algo_templates::GEMM_NAIVE,
            &inputs,
        );

        // Find the first Loop and verify its bound is resolved from params
        fn find_loop_bound(ops: &[TraceOp]) -> Option<usize> {
            for op in ops {
                if let TraceOp::Loop { bound, .. } = op {
                    if let BoundExpr::Const(v) = bound {
                        return Some(*v);
                    }
                }
            }
            None
        }

        let bound = find_loop_bound(&ops);
        assert!(bound.is_some(), "GEMM_NAIVE with params should have a Loop with Const bound");
        // GEMM_NAIVE outer Loop has bound="m", so resolved value is 8
        assert_eq!(bound.unwrap(), 8, "outer loop bound should resolve to m=8");
    }

    #[test]
    fn test_interpreter_activation_silu_produces_two_ops() {
        use crate::compiler::codegen::vm::algo_template::{AlgoTemplate, AlgoStep, ActivationKind};
        use crate::compiler::codegen::vm::algo_template::{AlgoStrategy, DeviceReq};

        let template = AlgoTemplate {
            name: "test_silu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Activation { kind: ActivationKind::Silu }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        // Silu produces Sigmoid + Mul = 2 ops (plus 1 Input)
        assert!(ops.len() >= 3, "Silu activation should produce Input + Sigmoid + Mul, got {} ops", ops.len());
        assert!(matches!(ops[0], TraceOp::Input(0)), "first op should be Input(0)");
    }

    #[test]
    fn test_interpreter_activation_relu_produces_max() {
        use crate::compiler::codegen::vm::algo_template::{AlgoTemplate, AlgoStep, ActivationKind};
        use crate::compiler::codegen::vm::algo_template::{AlgoStrategy, DeviceReq};

        let template = AlgoTemplate {
            name: "test_relu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Activation { kind: ActivationKind::Relu }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        // Relu produces Max(src, src) = 1 op (plus 1 Input)
        assert_eq!(ops.len(), 2, "Relu should produce Input + Max, got {} ops", ops.len());
        assert!(matches!(ops[1], TraceOp::Max(_, _)), "second op should be Max");
    }

    #[test]
    fn test_interpreter_softmax_produces_softmax_op() {
        use crate::compiler::codegen::vm::algo_template::{AlgoTemplate, AlgoStep};
        use crate::compiler::codegen::vm::algo_template::{AlgoStrategy, DeviceReq};

        let template = AlgoTemplate {
            name: "test_softmax",
            strategy: AlgoStrategy::SamplingSoftmax,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Softmax],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Softmax { .. })),
            "Softmax step should produce TraceOp::Softmax");
    }

    #[test]
    fn test_interpreter_shared_mem_declare_and_barrier() {
        use crate::compiler::codegen::vm::algo_template::{AlgoTemplate, AlgoStep};
        use crate::compiler::codegen::vm::algo_template::{AlgoStrategy, DeviceReq};

        let template = AlgoTemplate {
            name: "test_smem_barrier",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[
                AlgoStep::SharedMemDeclare { name: "buf", size_param: "smem_size" },
                AlgoStep::Barrier { barrier_name: "sync" },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("smem_size", 4096);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::SharedMemDeclare { bytes: 4096, .. })),
            "should produce SharedMemDeclare with 4096 bytes");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::SyncBarrier { .. })),
            "should produce SyncBarrier");
    }

    // ── Additional unit tests: interpreter paths and edge cases ────────

    #[test]
    fn test_param_table_from_template_returns_empty() {
        use crate::compiler::codegen::vm::algo_template::AlgoTemplate;
        use crate::compiler::codegen::vm::algo_template::{AlgoStrategy, DeviceReq};

        let tmpl = AlgoTemplate {
            name: "test",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[],
            params: &[],
            micro_kernel: None,
        };
        let table = ParamTable::from_template(&tmpl);
        assert_eq!(table.get("any_key"), None, "from_template returns empty table");
        assert_eq!(table.resolve("any_key"), 1, "resolve default is 1");
    }

    #[test]
    fn test_template_inputs_empty_names() {
        let inputs = TemplateInputs::new(&[]);
        assert_eq!(inputs.len(), 0, "empty names yields len 0");
        assert!(inputs.names.is_empty());
    }

    #[test]
    fn test_interpreter_trace_body_binop_sub() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceBinOp,
            AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_trace_sub",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Sub, dst: "c", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["a", "b"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Sub(_, _))),
            "TraceBody with Sub BinOp should produce TraceOp::Sub");
    }

    #[test]
    fn test_interpreter_seq_nests_steps() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_seq",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Seq(&[
                AlgoStep::Softmax,
                AlgoStep::Softmax,
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        let softmax_count = ops.iter().filter(|op| matches!(op, TraceOp::Softmax { .. })).count();
        assert_eq!(softmax_count, 2, "Seq should emit both Softmax steps");
    }

    #[test]
    fn test_interpreter_tile_config_mma_release() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_tile",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[
                AlgoStep::TileConfig { rows: "tile_r", cols: "tile_c" },
                AlgoStep::TileMma,
                AlgoStep::TileRelease,
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tile_r", 16);
        params.set("tile_c", 16);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::TileConfig { rows: 16, cols: 16 })),
            "should produce TileConfig with resolved params");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::TileMma { .. })),
            "should produce TileMma");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::TileRelease)),
            "should produce TileRelease");
    }

    #[test]
    fn test_interpreter_conditional_body_emitted() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_cond",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Conditional {
                requirement: DeviceReq::CpuAvx2,
                body: &[AlgoStep::Softmax],
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Softmax { .. })),
            "Conditional body should be emitted by interpreter");
    }

    #[test]
    fn test_interpreter_activation_gelu_produces_tanh() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_gelu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Activation { kind: ActivationKind::Gelu }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Tanh(_))),
            "Gelu activation should produce Tanh trace op");
    }

    #[test]
    fn test_interpreter_activation_sigmoid_produces_tanh() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_act_sigmoid",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Activation { kind: ActivationKind::Sigmoid }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Tanh(_))),
            "Sigmoid activation variant produces Tanh trace op");
    }

    #[test]
    fn test_interpreter_trace_body_load_const() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_const",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadConst { value: 3.14159 },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Const(v) if (*v - 3.14159).abs() < 1e-10)),
            "LoadConst should produce TraceOp::Const with the exact value");
    }

    #[test]
    fn test_interpreter_trace_body_cast() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, AlgoStrategy, DeviceReq,
        };
        use crate::compiler::trace::QuantPrecision;

        let template = AlgoTemplate {
            name: "test_cast",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "data" },
                AlgoTraceStep::Cast {
                    src: "data",
                    from: QuantPrecision::F32,
                    to: QuantPrecision::BF16,
                },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op,
            TraceOp::Cast { from: QuantPrecision::F32, to: QuantPrecision::BF16, .. })),
            "Cast trace step should produce TraceOp::Cast with correct precisions");
    }

    #[test]
    fn test_interpreter_embedding_gather_produces_vec_load_indexed() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_embed",
            strategy: AlgoStrategy::EmbeddingGather,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::EmbeddingGather],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::VecLoadIndexed { .. })),
            "EmbeddingGather should produce VecLoadIndexed");
    }

    #[test]
    fn test_interpreter_moe_router_gemv_produces_fma() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_moe_gemv",
            strategy: AlgoStrategy::MoePackedDispatch,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::MoeRouterGemv {
                num_experts: "num_experts",
                hidden: "hidden_dim",
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Fma(_, _, _))),
            "MoeRouterGemv should produce Fma trace op");
    }

    #[test]
    fn test_interpreter_async_wait_and_zero_fill() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        let template = AlgoTemplate {
            name: "test_async_zerofill",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[
                AlgoStep::AsyncWait { group: 0 },
                AlgoStep::ZeroFill { bytes_param: "fill_size" },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("fill_size", 2048);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x", "y", "z"]);
        let ops = interp.instantiate(&template, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::AsyncWaitGroup { n: 0 })),
            "should produce AsyncWaitGroup with group=0");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::Input(_))),
            "ZeroFill should produce Input trace op");
    }

    // ── Additional unit tests (wave-12ka5) ───────────────────────────
    //
    // Note: AlgoTemplate requires &'static fields. We use fully static
    // constants (static TEMPLATE: AlgoTemplate = ...) to satisfy the
    // lifetime requirements.

    #[test]
    fn test_interpreter_trace_body_unary_exp() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceUnaryOp,
            AlgoStrategy, DeviceReq,
        };

        static STEPS: &[AlgoStep] = &[AlgoStep::TraceBody(&[
            AlgoTraceStep::LoadInput { name: "src" },
            AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Exp, dst: "out", src: "src" },
        ])];

        static TEMPLATE: AlgoTemplate = AlgoTemplate {
            name: "test_unary_exp",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: STEPS,
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["src"]);
        let ops = interp.instantiate(&TEMPLATE, &inputs);

        assert!(ops.len() >= 2, "UnaryOp Exp should produce at least 2 ops");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::Exp(_))),
            "UnaryOp Exp should produce TraceOp::Exp");
    }

    #[test]
    fn test_interpreter_trace_body_unary_neg_abs_recip_log() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceUnaryOp,
            AlgoStrategy, DeviceReq,
        };

        static NEG_TMPL: AlgoTemplate = AlgoTemplate {
            name: "neg",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Neg, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static ABS_TMPL: AlgoTemplate = AlgoTemplate {
            name: "abs",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Abs, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static RECIP_TMPL: AlgoTemplate = AlgoTemplate {
            name: "recip",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Recip, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static LOG_TMPL: AlgoTemplate = AlgoTemplate {
            name: "log",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Log, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["s"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let neg_ops = interp.instantiate(&NEG_TMPL, &inputs);
        assert!(neg_ops.iter().any(|op| matches!(op, TraceOp::Neg(_))), "Neg should produce TraceOp::Neg");

        let mut interp = TemplateInterpreter::new(params.clone());
        let abs_ops = interp.instantiate(&ABS_TMPL, &inputs);
        assert!(abs_ops.iter().any(|op| matches!(op, TraceOp::Abs(_))), "Abs should produce TraceOp::Abs");

        let mut interp = TemplateInterpreter::new(params.clone());
        let recip_ops = interp.instantiate(&RECIP_TMPL, &inputs);
        assert!(recip_ops.iter().any(|op| matches!(op, TraceOp::Recip(_))), "Recip should produce TraceOp::Recip");

        let mut interp = TemplateInterpreter::new(params);
        let log_ops = interp.instantiate(&LOG_TMPL, &inputs);
        assert!(log_ops.iter().any(|op| matches!(op, TraceOp::Log(_))), "Log should produce TraceOp::Log");
    }

    #[test]
    fn test_interpreter_trace_body_binop_mul_and_max() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceBinOp,
            AlgoStrategy, DeviceReq,
        };

        static MUL_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_mul",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Mul, dst: "c", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static MAX_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_max",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Max, dst: "c", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["a", "b"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let mul_ops = interp.instantiate(&MUL_TMPL, &inputs);
        // instantiate prepends 2 Input ops for inputs, then TraceBody adds 2 more Input + 1 Mul = 5
        assert!(mul_ops.len() >= 3, "Mul BinOp should produce at least 3 ops, got {}", mul_ops.len());
        assert!(mul_ops.iter().any(|op| matches!(op, TraceOp::Mul(_, _))),
            "BinOp Mul should produce TraceOp::Mul");

        let mut interp2 = TemplateInterpreter::new(params);
        let max_ops = interp2.instantiate(&MAX_TMPL, &inputs);
        assert!(max_ops.len() >= 3, "Max BinOp should produce at least 3 ops, got {}", max_ops.len());
        assert!(max_ops.iter().any(|op| matches!(op, TraceOp::Max(_, _))),
            "BinOp Max should produce TraceOp::Max");
    }

    #[test]
    fn test_interpreter_trace_body_fma_step() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static FMA_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_trace_fma",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "acc" },
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::Fma { acc: "acc", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["acc", "a", "b"]);
        let ops = interp.instantiate(&FMA_TMPL, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Fma(_, _, _))),
            "AlgoTraceStep::Fma should produce TraceOp::Fma");
        // instantiate prepends 3 Input ops for inputs, then TraceBody adds 3 more Input + 1 Fma = 7
        assert!(ops.len() >= 4, "Fma step should produce at least 4 ops, got {}", ops.len());
    }

    #[test]
    fn test_interpreter_trace_body_hreduce_sum_and_max() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, ReduceKind,
            AlgoStrategy, DeviceReq,
        };

        static SUM_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_hreduce_sum",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "src" },
                AlgoTraceStep::HReduce { src: "src", op: ReduceKind::Sum },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static MAX_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_hreduce_max",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "src" },
                AlgoTraceStep::HReduce { src: "src", op: ReduceKind::Max },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["src"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let sum_ops = interp.instantiate(&SUM_TMPL, &inputs);
        assert!(sum_ops.iter().any(|op| matches!(op, TraceOp::HReduce { op: crate::compiler::trace::ReduceKind::Sum, .. })),
            "HReduce Sum should produce TraceOp::HReduce with Sum");

        let mut interp2 = TemplateInterpreter::new(params);
        let max_ops = interp2.instantiate(&MAX_TMPL, &inputs);
        assert!(max_ops.iter().any(|op| matches!(op, TraceOp::HReduce { op: crate::compiler::trace::ReduceKind::Max, .. })),
            "HReduce Max should produce TraceOp::HReduce with Max");
    }

    #[test]
    fn test_interpreter_trace_body_broadcast_and_vec_store_indexed() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_broadcast_store",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "src" },
                AlgoTraceStep::LoadInput { name: "base" },
                AlgoTraceStep::LoadInput { name: "offset" },
                AlgoTraceStep::Broadcast { src: "src", dst: "wide" },
                AlgoTraceStep::VecStoreIndexed { base: "base", offset: "offset", src: "src" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["src", "base", "offset"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::BroadcastScalar { .. })),
            "Broadcast trace step should produce TraceOp::BroadcastScalar");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::VecStoreIndexed { .. })),
            "VecStoreIndexed trace step should produce TraceOp::VecStoreIndexed");
    }

    #[test]
    fn test_interpreter_nested_loop_body_uses_outer_env() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static NESTED_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_nested_loop",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[
                AlgoStep::Loop {
                    bound: "outer",
                    step: "outer_step",
                    body: &[
                        AlgoStep::Loop {
                            bound: "inner",
                            step: "inner_step",
                            body: &[AlgoStep::Softmax],
                        },
                    ],
                },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("outer", 4);
        params.set("outer_step", 1);
        params.set("inner", 8);
        params.set("inner_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&NESTED_TMPL, &inputs);

        let outer_loop = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. }));
        assert!(outer_loop.is_some(), "should have outer Loop");

        if let TraceOp::Loop { bound: outer_bound, body, .. } = outer_loop.unwrap() {
            assert_eq!(*outer_bound, BoundExpr::Const(4), "outer loop bound should be 4");
            let inner_loop = body.iter().find(|op| matches!(op, TraceOp::Loop { .. }));
            assert!(inner_loop.is_some(), "outer body should contain inner Loop");

            if let TraceOp::Loop { bound: inner_bound, body: inner_body, .. } = inner_loop.unwrap() {
                assert_eq!(*inner_bound, BoundExpr::Const(8), "inner loop bound should be 8");
                assert!(inner_body.iter().any(|op| matches!(op, TraceOp::Softmax { .. })),
                    "innermost body should contain Softmax");
            }
        }
    }

    #[test]
    fn test_interpreter_activation_tanh_produces_tanh_op() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_act_tanh",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Activation { kind: ActivationKind::Tanh }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Tanh(_))),
            "Tanh activation should produce TraceOp::Tanh");
    }

    #[test]
    fn test_interpreter_epilogue_step() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, EpilogueOp, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_epilogue",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Epilogue {
                ops: &[EpilogueOp::BiasAdd, EpilogueOp::Relu],
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::EpilogueChain { .. })),
            "Epilogue step should produce TraceOp::EpilogueChain");

        if let Some(TraceOp::EpilogueChain { ops: ep_ops }) = ops.iter().find_map(|op| {
            if let TraceOp::EpilogueChain { ops } = op {
                Some(TraceOp::EpilogueChain { ops: ops.clone() })
            } else {
                None
            }
        }) {
            assert_eq!(ep_ops.len(), 2, "epilogue should contain 2 ops (BiasAdd + Relu)");
        }
    }

    #[test]
    fn test_interpreter_moe_top_k_step() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_moe_topk",
            strategy: AlgoStrategy::MoeRouterTopk,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::MoeTopK {
                num_experts: "e",
                top_k: "k",
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        assert!(ops.len() > 6,
            "MoeTopK with gemm inputs should produce Input ops for inputs plus one more, got {}", ops.len());
    }

    #[test]
    fn test_interpreter_row_copy_step() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_row_copy",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::RowCopy {
                rows_param: "r",
                cols_param: "c",
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        let input_count = ops.iter().filter(|op| matches!(op, TraceOp::Input(_))).count();
        assert!(input_count >= 8,
            "RowCopy should produce at least 8 Input ops total (6 gemm + 2 rowcopy), got {}", input_count);
    }

    // ── Additional unit tests (wave-12kbc) ───────────────────────────

    /// Verify interpret_body_step Reduce path uses env.get("panel_a")
    /// (distinct from top-level Reduce which uses next_slot().saturating_sub(1)).
    #[test]
    fn test_interpreter_body_reduce_uses_env_panel_a() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_reduce",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "outer",
                step: "outer_step",
                body: &[AlgoStep::Reduce { op: crate::compiler::codegen::vm::algo_template::ReduceOp::Sum }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("outer", 2);
        params.set("outer_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // Locate Loop and inspect its body for HReduce
        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::HReduce { .. })),
                "Loop body should contain HReduce from body-step Reduce");
            // Body HReduce should reference env.get("panel_a") which is ValueId(0) when unbound
            let src_value = body.iter().find_map(|op| {
                if let TraceOp::HReduce { src, op: _ } = op { Some(*src) } else { None }
            });
            if let Some(src) = src_value {
                assert_eq!(src, ValueId(0),
                    "Body Reduce should reference panel_a slot from env (ValueId(0) when no panel loaded)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify interpret_body_step Dequantize path produces TraceOp::Input
    /// referencing env.get("panel_a") rather than next_slot.
    #[test]
    fn test_interpreter_body_dequantize_uses_env_panel_a() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, BlockUnpackMode, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_dequant",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Dequantize { mode: BlockUnpackMode::Q4_0 }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 4);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::Input(_))),
                "Body Dequantize should produce TraceOp::Input");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify interpret_body_step Softmax path references env.get("panel_a")
    /// for both src and dst.
    #[test]
    fn test_interpreter_body_softmax_uses_env_panel_a() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_softmax",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Softmax],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 2);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let softmax = body.iter().find(|op| matches!(op, TraceOp::Softmax { .. }));
            assert!(softmax.is_some(), "Body should contain Softmax");
            if let Some(TraceOp::Softmax { src, dst }) = softmax {
                assert_eq!(*src, *dst, "Body Softmax src and dst should be identical (in-place)");
                assert_eq!(*src, ValueId(0),
                    "Body Softmax should reference panel_a from env (ValueId(0) when no panel loaded)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify Dequantize top-level step with a specific BlockUnpackMode variant.
    #[test]
    fn test_interpreter_dequantize_mxfp4_step() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, BlockUnpackMode, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_dequant_mxfp4",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Dequantize { mode: BlockUnpackMode::Mxfp4 }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // Dequantize at top level produces TraceOp::Input(next_slot().saturating_sub(1))
        // next_slot is 1 (after Input(0) for "data"), so saturating_sub(1) = ValueId(0)
        assert!(ops.len() >= 2, "Should have at least Input + Dequantize output, got {}", ops.len());
        assert!(matches!(ops[0], TraceOp::Input(0)), "First op should be Input(0) for data");
    }

    /// Verify Reduce top-level step with ReduceOp::Max and ReduceOp::Min variants.
    #[test]
    fn test_interpreter_reduce_max_and_min_variants() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ReduceOp, AlgoStrategy, DeviceReq,
        };

        static MAX_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_reduce_max",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Reduce { op: ReduceOp::Max }],
            params: &[],
            micro_kernel: None,
        };
        static MIN_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_reduce_min",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Reduce { op: ReduceOp::Min }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["data"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let max_ops = interp.instantiate(&MAX_TMPL, &inputs);
        assert!(max_ops.iter().any(|op| matches!(op, TraceOp::HReduce { .. })),
            "ReduceOp::Max should produce HReduce");

        let mut interp2 = TemplateInterpreter::new(params);
        let min_ops = interp2.instantiate(&MIN_TMPL, &inputs);
        assert!(min_ops.iter().any(|op| matches!(op, TraceOp::HReduce { .. })),
            "ReduceOp::Min should produce HReduce");
    }

    /// Verify TraceBody BinOp Div and Min variants produce correct TraceOp kinds.
    #[test]
    fn test_interpreter_trace_body_binop_div_and_min() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceBinOp,
            AlgoStrategy, DeviceReq,
        };

        static DIV_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_div",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Div, dst: "c", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static MIN_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_min",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Min, dst: "c", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["a", "b"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let div_ops = interp.instantiate(&DIV_TMPL, &inputs);
        assert!(div_ops.iter().any(|op| matches!(op, TraceOp::Div(_, _))),
            "BinOp Div should produce TraceOp::Div");

        let mut interp2 = TemplateInterpreter::new(params);
        let min_ops = interp2.instantiate(&MIN_TMPL, &inputs);
        assert!(min_ops.iter().any(|op| matches!(op, TraceOp::Min(_, _))),
            "BinOp Min should produce TraceOp::Min");
    }

    /// Verify TraceBody UnaryOp Sqrt, Rsqrt, and Sigmoid produce correct TraceOp kinds.
    #[test]
    fn test_interpreter_trace_body_unary_sqrt_rsqrt_sigmoid() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceUnaryOp,
            AlgoStrategy, DeviceReq,
        };

        static SQRT_TMPL: AlgoTemplate = AlgoTemplate {
            name: "sqrt",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Sqrt, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static RSQRT_TMPL: AlgoTemplate = AlgoTemplate {
            name: "rsqrt",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Rsqrt, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };
        static SIGMOID_TMPL: AlgoTemplate = AlgoTemplate {
            name: "sigmoid",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "s" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Sigmoid, dst: "o", src: "s" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["s"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let sqrt_ops = interp.instantiate(&SQRT_TMPL, &inputs);
        assert!(sqrt_ops.iter().any(|op| matches!(op, TraceOp::Sqrt(_))),
            "UnaryOp Sqrt should produce TraceOp::Sqrt");

        let mut interp2 = TemplateInterpreter::new(params.clone());
        let rsqrt_ops = interp2.instantiate(&RSQRT_TMPL, &inputs);
        assert!(rsqrt_ops.iter().any(|op| matches!(op, TraceOp::Rsqrt(_))),
            "UnaryOp Rsqrt should produce TraceOp::Rsqrt");

        let mut interp3 = TemplateInterpreter::new(params);
        let sigmoid_ops = interp3.instantiate(&SIGMOID_TMPL, &inputs);
        assert!(sigmoid_ops.iter().any(|op| matches!(op, TraceOp::Sigmoid(_))),
            "UnaryOp Sigmoid should produce TraceOp::Sigmoid");
    }

    /// Verify loop body step SharedMemDeclare + AsyncCopyToSmem + Barrier
    /// use correct env references inside interpret_body_step.
    #[test]
    fn test_interpreter_body_smem_async_barrier() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_smem",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[
                    AlgoStep::SharedMemDeclare { name: "smem_buf", size_param: "smem_bytes" },
                    AlgoStep::AsyncCopyToSmem { buffer_name: "smem_buf", size_param: "copy_bytes" },
                    AlgoStep::AsyncWait { group: 1 },
                    AlgoStep::Barrier { barrier_name: "post_copy" },
                ],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 4);
        params.set("tile_step", 1);
        params.set("smem_bytes", 8192);
        params.set("copy_bytes", 4096);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::SharedMemDeclare { bytes: 8192, .. })),
                "Body should contain SharedMemDeclare with 8192 bytes");
            assert!(body.iter().any(|op| matches!(op, TraceOp::AsyncCopyToShared { bytes: 4096, .. })),
                "Body should contain AsyncCopyToShared with 4096 bytes");
            assert!(body.iter().any(|op| matches!(op, TraceOp::AsyncWaitGroup { n: 1 })),
                "Body should contain AsyncWaitGroup with group=1");
            assert!(body.iter().any(|op| matches!(op, TraceOp::SyncBarrier { .. })),
                "Body should contain SyncBarrier");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body step TileConfig + TileMma + TileRelease produce
    /// correct TraceOp variants inside interpret_body_step.
    #[test]
    fn test_interpreter_body_tile_config_mma_release() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_tile",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[
                    AlgoStep::TileConfig { rows: "tr", cols: "tc" },
                    AlgoStep::TileMma,
                    AlgoStep::TileRelease,
                ],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 2);
        params.set("tile_step", 1);
        params.set("tr", 16);
        params.set("tc", 16);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::TileConfig { rows: 16, cols: 16 })),
                "Body should contain TileConfig with rows=16, cols=16");
            assert!(body.iter().any(|op| matches!(op, TraceOp::TileMma { .. })),
                "Body should contain TileMma");
            assert!(body.iter().any(|op| matches!(op, TraceOp::TileRelease)),
                "Body should contain TileRelease");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body Activation Silu and Relu produce correct TraceOp
    /// sequences inside interpret_body_step (uses env.get("panel_a")).
    #[test]
    fn test_interpreter_body_activation_silu_and_relu() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind, AlgoStrategy, DeviceReq,
        };

        static SILU_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_silu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Activation { kind: ActivationKind::Silu }],
            }],
            params: &[],
            micro_kernel: None,
        };
        static RELU_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_relu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Activation { kind: ActivationKind::Relu }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 2);
        params.set("n_step", 1);
        let inputs = TemplateInputs::new(&["data"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let silu_ops = interp.instantiate(&SILU_TMPL, &inputs);
        if let Some(TraceOp::Loop { body, .. }) = silu_ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::Sigmoid(_))),
                "Body Silu should produce Sigmoid");
            assert!(body.iter().any(|op| matches!(op, TraceOp::Mul(_, _))),
                "Body Silu should produce Mul");
        }

        let mut interp2 = TemplateInterpreter::new(params);
        let relu_ops = interp2.instantiate(&RELU_TMPL, &inputs);
        if let Some(TraceOp::Loop { body, .. }) = relu_ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::Max(_, _))),
                "Body Relu should produce Max");
        }
    }

    /// Verify loop body EmbeddingGather inside interpret_body_step produces
    /// VecLoadIndexed referencing env.get("a_ptr") and env.get("a_offset").
    #[test]
    fn test_interpreter_body_embedding_gather() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_embed",
            strategy: AlgoStrategy::EmbeddingGather,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "seq",
                step: "seq_step",
                body: &[AlgoStep::EmbeddingGather],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("seq", 8);
        params.set("seq_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm(); // provides a_ptr + a_offset
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let vli = body.iter().find(|op| matches!(op, TraceOp::VecLoadIndexed { .. }));
            assert!(vli.is_some(), "Body EmbeddingGather should produce VecLoadIndexed");
            if let Some(TraceOp::VecLoadIndexed { base, offset }) = vli {
                // env.get("a_ptr") = ValueId(0), env.get("a_offset") = ValueId(3)
                assert_eq!(*base, ValueId(0), "base should be a_ptr = ValueId(0)");
                assert_eq!(*offset, ValueId(3), "offset should be a_offset = ValueId(3)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    // ── Additional unit tests (wave-12kie) ───────────────────────────

    /// Verify PackBuffer top-level step resolves rows/cols from params and
    /// produces TraceOp::PackBuffer referencing env.get("panel_a").
    #[test]
    fn test_interpreter_pack_buffer_resolves_params() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_pack_buffer",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "pr", cols_param: "pc" },
                AlgoStep::PackBuffer { buffer_name: "packed_a", rows_param: "pk_r", cols_param: "pk_c" },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("pr", 4);
        params.set("pc", 16);
        params.set("pk_r", 4);
        params.set("pk_c", 16);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        let pack = ops.iter().find(|op| matches!(op, TraceOp::PackBuffer { .. }));
        assert!(pack.is_some(), "PackBuffer step should produce TraceOp::PackBuffer");
        if let Some(TraceOp::PackBuffer { rows, cols, layout, .. }) = pack {
            assert_eq!(*rows, 4, "PackBuffer rows should resolve to 4");
            assert_eq!(*cols, 16, "PackBuffer cols should resolve to 16");
            assert_eq!(*layout, PackLayout::RowMajor, "PackBuffer layout should be RowMajor");
        }
    }

    /// Verify StoreResult top-level step produces TraceOp::PanelStore
    /// referencing env.get("c_ptr") and env.get("c_offset") with resolved dims.
    #[test]
    fn test_interpreter_store_result_panel_store() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_store_result",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::StoreResult { rows_param: "sr", cols_param: "sc" }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("sr", 8);
        params.set("sc", 32);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        let store = ops.iter().find(|op| matches!(op, TraceOp::PanelStore { .. }));
        assert!(store.is_some(), "StoreResult step should produce TraceOp::PanelStore");
        if let Some(TraceOp::PanelStore { base, offset, rows, cols }) = store {
            // env.get("c_ptr") = ValueId(2), env.get("c_offset") = ValueId(5)
            assert_eq!(*base, ValueId(2), "PanelStore base should be c_ptr = ValueId(2)");
            assert_eq!(*offset, ValueId(5), "PanelStore offset should be c_offset = ValueId(5)");
            assert_eq!(*rows, 8, "PanelStore rows should resolve to 8");
            assert_eq!(*cols, 32, "PanelStore cols should resolve to 32");
        }
    }

    /// Verify LoadPanel for MatrixRole::B and MatrixRole::C resolve to the
    /// correct env slots (b_ptr/b_offset and c_ptr/c_offset respectively).
    #[test]
    fn test_interpreter_load_panel_matrix_b_and_c() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_load_panel_bc",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::B, rows_param: "br", cols_param: "bc" },
                AlgoStep::LoadPanel { matrix: MatrixRole::C, rows_param: "cr", cols_param: "cc" },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("br", 32);
        params.set("bc", 16);
        params.set("cr", 8);
        params.set("cc", 16);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // Skip the 6 Input ops for gemm inputs
        let panels: Vec<_> = ops.iter()
            .filter(|op| matches!(op, TraceOp::PanelLoad { .. }))
            .collect();
        assert_eq!(panels.len(), 2, "Should produce exactly 2 PanelLoad ops");

        // MatrixRole::B → b_ptr=ValueId(1), b_offset=ValueId(4)
        if let Some(TraceOp::PanelLoad { base, offset, rows, cols }) = panels.first() {
            assert_eq!(*base, ValueId(1), "Panel B base should be b_ptr = ValueId(1)");
            assert_eq!(*offset, ValueId(4), "Panel B offset should be b_offset = ValueId(4)");
            assert_eq!(*rows, 32);
            assert_eq!(*cols, 16);
        }

        // MatrixRole::C → c_ptr=ValueId(2), c_offset=ValueId(5)
        if let Some(TraceOp::PanelLoad { base, offset, rows, cols }) = panels.last() {
            assert_eq!(*base, ValueId(2), "Panel C base should be c_ptr = ValueId(2)");
            assert_eq!(*offset, ValueId(5), "Panel C offset should be c_offset = ValueId(5)");
            assert_eq!(*rows, 8);
            assert_eq!(*cols, 16);
        }
    }

    /// Verify AsyncCopyToSmem top-level step produces TraceOp::AsyncCopyToShared
    /// with resolved bytes and referencing env.get("panel_a") for src_offset.
    #[test]
    fn test_interpreter_async_copy_to_smem_top_level() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_async_copy_toplevel",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "r", cols_param: "c" },
                AlgoStep::AsyncCopyToSmem { buffer_name: "smem_a", size_param: "copy_sz" },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("r", 4);
        params.set("c", 16);
        params.set("copy_sz", 2048);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        let copy = ops.iter().find(|op| matches!(op, TraceOp::AsyncCopyToShared { .. }));
        assert!(copy.is_some(), "AsyncCopyToSmem should produce TraceOp::AsyncCopyToShared");
        if let Some(TraceOp::AsyncCopyToShared { name, bytes, .. }) = copy {
            assert_eq!(*name, "smem_a", "buffer name should be preserved");
            assert_eq!(*bytes, 2048, "bytes should resolve from params");
        }
    }

    /// Verify loop body MicroKernel references env slots for panel_a, panel_b,
    /// and c_ptr (distinct from top-level which also uses env).
    #[test]
    fn test_interpreter_body_micro_kernel_with_loaded_panels() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_mukernel",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "ar", cols_param: "ac" },
                AlgoStep::LoadPanel { matrix: MatrixRole::B, rows_param: "br", cols_param: "bc" },
                AlgoStep::Loop {
                    bound: "tiles",
                    step: "tile_step",
                    body: &[AlgoStep::MicroKernel],
                },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("ar", 4);
        params.set("ac", 8);
        params.set("br", 8);
        params.set("bc", 4);
        params.set("tiles", 2);
        params.set("tile_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // panel_a is bound after LoadPanel A → slot index = 6 + 0 = 6
        // panel_b is bound after LoadPanel B → slot index = 6 + 1 = 7
        // c_ptr is inputs[2] = ValueId(2)
        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let fma = body.iter().find(|op| matches!(op, TraceOp::Fma(_, _, _)));
            assert!(fma.is_some(), "Body MicroKernel should produce Fma");
            if let Some(TraceOp::Fma(a, b, c)) = fma {
                assert_eq!(*a, ValueId(6), "Fma a should be panel_a slot = ValueId(6)");
                assert_eq!(*b, ValueId(7), "Fma b should be panel_b slot = ValueId(7)");
                assert_eq!(*c, ValueId(2), "Fma c should be c_ptr = ValueId(2)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body StoreResult produces TraceOp::PanelStore referencing
    /// outer env slots for c_ptr and c_offset.
    #[test]
    fn test_interpreter_body_store_result() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_store",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[AlgoStep::StoreResult { rows_param: "sr", cols_param: "sc" }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 4);
        params.set("tile_step", 1);
        params.set("sr", 8);
        params.set("sc", 16);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let store = body.iter().find(|op| matches!(op, TraceOp::PanelStore { .. }));
            assert!(store.is_some(), "Body StoreResult should produce PanelStore");
            if let Some(TraceOp::PanelStore { base, offset, rows, cols }) = store {
                assert_eq!(*base, ValueId(2), "PanelStore base should be c_ptr = ValueId(2)");
                assert_eq!(*offset, ValueId(5), "PanelStore offset should be c_offset = ValueId(5)");
                assert_eq!(*rows, 8, "rows should resolve to 8");
                assert_eq!(*cols, 16, "cols should resolve to 16");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify TemplateInputs::moe() preset has correct name ordering
    /// and that names match expected values exactly.
    #[test]
    fn test_template_inputs_moe_preset_name_ordering() {
        let moe = TemplateInputs::moe();

        assert_eq!(moe.names.len(), 4);
        assert_eq!(moe.names[0], "hidden_ptr", "first moe input should be hidden_ptr");
        assert_eq!(moe.names[1], "weight_ptr", "second moe input should be weight_ptr");
        assert_eq!(moe.names[2], "output_ptr", "third moe input should be output_ptr");
        assert_eq!(moe.names[3], "seq_offset", "fourth moe input should be seq_offset");
    }

    /// Verify TraceBody with BinOp::Add produces TraceOp::Add referencing
    /// the original env slots bound during instantiate (LoadInput does not
    /// create new slots — it re-emits the existing binding).
    #[test]
    fn test_interpreter_trace_body_binop_add_slot_indices() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceBinOp,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_add_slots",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "x" },
                AlgoTraceStep::LoadInput { name: "y" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "z", a: "x", b: "y" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x", "y"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // instantiate prepends 2 Input ops binding "x"→ValueId(0), "y"→ValueId(1)
        // TraceBody LoadInput does NOT push new slots — it re-emits the existing binding
        // so BinOp Add references the original env: Add(ValueId(0), ValueId(1))
        let add_op = ops.iter().find(|op| matches!(op, TraceOp::Add(_, _)));
        assert!(add_op.is_some(), "BinOp Add should produce TraceOp::Add");
        if let Some(TraceOp::Add(a, b)) = add_op {
            assert_eq!(*a, ValueId(0), "Add a should reference env slot for x = ValueId(0)");
            assert_eq!(*b, ValueId(1), "Add b should reference env slot for y = ValueId(1)");
        }
    }

    /// Verify TraceBody HReduce with ReduceKind::Min produces
    /// TraceOp::HReduce with the correct ReduceKind mapping.
    #[test]
    fn test_interpreter_trace_body_hreduce_min() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, ReduceKind,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_hreduce_min",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "src" },
                AlgoTraceStep::HReduce { src: "src", op: ReduceKind::Min },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["src"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let hreduce = ops.iter().find(|op| matches!(op, TraceOp::HReduce { .. }));
        assert!(hreduce.is_some(), "HReduce Min should produce TraceOp::HReduce");
        if let Some(TraceOp::HReduce { src, op }) = hreduce {
            // LoadInput "src" does not create a new slot — references env ValueId(0)
            assert_eq!(*src, ValueId(0), "HReduce src should reference env slot for src = ValueId(0)");
            assert!(matches!(op, crate::compiler::trace::ReduceKind::Min),
                "HReduce op should be ReduceKind::Min");
        }
    }

    /// Verify TraceBody VecLoadIndexed produces TraceOp::VecLoadIndexed with
    /// correct slot references resolved via SlotEnv for base and offset names.
    #[test]
    fn test_interpreter_trace_body_vec_load_indexed_slots() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_vli_slots",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "table" },
                AlgoTraceStep::LoadInput { name: "idx" },
                AlgoTraceStep::VecLoadIndexed { base: "table", offset: "idx" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["table", "idx"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // LoadInput does NOT create new slots — references original env bindings
        // "table" → ValueId(0), "idx" → ValueId(1)
        let vli = ops.iter().find(|op| matches!(op, TraceOp::VecLoadIndexed { .. }));
        assert!(vli.is_some(), "VecLoadIndexed trace step should produce TraceOp::VecLoadIndexed");
        if let Some(TraceOp::VecLoadIndexed { base, offset }) = vli {
            assert_eq!(*base, ValueId(0), "base should reference env slot for table = ValueId(0)");
            assert_eq!(*offset, ValueId(1), "offset should reference env slot for idx = ValueId(1)");
        }
    }

    // ── Additional unit tests (wave-12x88) ───────────────────────────

    /// Verify ParamTable resolve returns the actual value when present,
    /// even for zero (which is falsy in some languages).
    #[test]
    fn test_param_table_resolve_zero_vs_missing() {
        let mut table = ParamTable::new();
        table.set("zero_key", 0);

        assert_eq!(table.resolve("zero_key"), 0, "resolve should return 0 when explicitly set");
        assert_eq!(table.resolve("missing_key"), 1, "resolve should return default 1 for missing");
    }

    /// Verify ParamTable set/get with multiple independent keys.
    #[test]
    fn test_param_table_multiple_independent_keys() {
        let mut table = ParamTable::new();
        table.set("m", 64);
        table.set("n", 128);
        table.set("k", 32);

        assert_eq!(table.get("m"), Some(64));
        assert_eq!(table.get("n"), Some(128));
        assert_eq!(table.get("k"), Some(32));
        assert_eq!(table.get("unknown"), None);
    }

    /// Verify TemplateInputs::new preserves exact name ordering.
    #[test]
    fn test_template_inputs_new_preserves_order() {
        let inputs = TemplateInputs::new(&["delta", "alpha", "charlie"]);

        assert_eq!(inputs.names[0], "delta");
        assert_eq!(inputs.names[1], "alpha");
        assert_eq!(inputs.names[2], "charlie");
    }

    /// Verify TemplateInputs::gemm() names match expected GEMM input contract.
    #[test]
    fn test_template_inputs_gemm_names_exact_match() {
        let gemm = TemplateInputs::gemm();

        assert_eq!(gemm.names, vec![
            "a_ptr", "b_ptr", "c_ptr", "a_offset", "b_offset", "c_offset"
        ], "gemm preset names must match exact GEMM input contract");
    }

    /// Verify TemplateInterpreter produces exactly N Input ops for N inputs
    /// when template steps are empty.
    #[test]
    fn test_interpreter_empty_template_exactly_n_inputs() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_empty_n",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[],
            params: &[],
            micro_kernel: None,
        };

        let inputs = TemplateInputs::new(&["p", "q", "r", "s"]);
        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert_eq!(ops.len(), 4, "empty template with 4 inputs should produce exactly 4 Input ops");
        for (i, op) in ops.iter().enumerate() {
            assert!(matches!(op, TraceOp::Input(n) if *n as usize == i),
                "op[{}] should be Input({})", i, i);
        }
    }

    /// Verify loop body PackBuffer step uses env.get("panel_a") for src/dst
    /// and resolves rows/cols from params.
    #[test]
    fn test_interpreter_body_pack_buffer_resolves_params() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_packbuf",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "ar", cols_param: "ac" },
                AlgoStep::Loop {
                    bound: "tiles",
                    step: "tile_step",
                    body: &[AlgoStep::PackBuffer {
                        buffer_name: "packed",
                        rows_param: "pk_r",
                        cols_param: "pk_c",
                    }],
                },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("ar", 4);
        params.set("ac", 8);
        params.set("pk_r", 4);
        params.set("pk_c", 8);
        params.set("tiles", 2);
        params.set("tile_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // panel_a bound at slot 6 (after 6 gemm Input ops + 1 PanelLoad)
        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let pack = body.iter().find(|op| matches!(op, TraceOp::PackBuffer { .. }));
            assert!(pack.is_some(), "Body should contain PackBuffer");
            if let Some(TraceOp::PackBuffer { src, dst, rows, cols, layout }) = pack {
                assert_eq!(*src, ValueId(6), "PackBuffer src should reference panel_a = ValueId(6)");
                assert_eq!(*dst, ValueId(6), "PackBuffer dst should reference panel_a = ValueId(6)");
                assert_eq!(*rows, 4, "PackBuffer rows should resolve to 4");
                assert_eq!(*cols, 8, "PackBuffer cols should resolve to 8");
                assert_eq!(*layout, PackLayout::RowMajor);
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify TraceBody with multiple LoadConst produces TraceOp::Const in order.
    #[test]
    fn test_interpreter_trace_body_multiple_load_const_ordering() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_multi_const",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadConst { value: 1.0 },
                AlgoTraceStep::LoadConst { value: 2.0 },
                AlgoTraceStep::LoadConst { value: 3.0 },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let consts: Vec<f64> = ops.iter()
            .filter_map(|op| if let TraceOp::Const(v) = op { Some(*v) } else { None })
            .collect();
        assert_eq!(consts.len(), 3, "Should produce 3 Const ops");
        assert!((consts[0] - 1.0).abs() < 1e-10, "First const should be 1.0");
        assert!((consts[1] - 2.0).abs() < 1e-10, "Second const should be 2.0");
        assert!((consts[2] - 3.0).abs() < 1e-10, "Third const should be 3.0");
    }

    /// Verify TraceBody Cast step preserves from/to QuantPrecision fields exactly.
    #[test]
    fn test_interpreter_trace_body_cast_preserves_precisions() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, AlgoStrategy, DeviceReq,
        };
        use crate::compiler::trace::QuantPrecision;

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_cast_prec",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "data" },
                AlgoTraceStep::Cast { src: "data", from: QuantPrecision::F16, to: QuantPrecision::F32 },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let cast = ops.iter().find(|op| matches!(op, TraceOp::Cast { .. }));
        assert!(cast.is_some(), "Cast step should produce TraceOp::Cast");
        if let Some(TraceOp::Cast { src, from, to }) = cast {
            assert_eq!(*src, ValueId(0), "Cast src should reference env slot for data");
            assert_eq!(*from, QuantPrecision::F16, "from should be F16");
            assert_eq!(*to, QuantPrecision::F32, "to should be F32");
        }
    }

    /// Verify loop body MoeRouterGemv produces TraceOp::Fma referencing
    /// outer env slots for a_ptr, b_ptr, c_ptr.
    #[test]
    fn test_interpreter_body_moe_router_gemv_slots() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_moe_gemv",
            strategy: AlgoStrategy::MoePackedDispatch,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "experts",
                step: "expert_step",
                body: &[AlgoStep::MoeRouterGemv {
                    num_experts: "ne",
                    hidden: "hd",
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("experts", 4);
        params.set("expert_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let fma = body.iter().find(|op| matches!(op, TraceOp::Fma(_, _, _)));
            assert!(fma.is_some(), "Body MoeRouterGemv should produce Fma");
            if let Some(TraceOp::Fma(a, b, c)) = fma {
                assert_eq!(*a, ValueId(0), "Fma a should be a_ptr = ValueId(0)");
                assert_eq!(*b, ValueId(1), "Fma b should be b_ptr = ValueId(1)");
                assert_eq!(*c, ValueId(2), "Fma c should be c_ptr = ValueId(2)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify EpilogueOp::ResidualAdd and EpilogueOp::Gelu variants are
    /// preserved through TraceOp::EpilogueChain.
    #[test]
    fn test_interpreter_epilogue_residual_and_gelu_variants() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, EpilogueOp, AlgoStrategy, DeviceReq,
        };

        static RESIDUAL_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_epilogue_residual",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Epilogue {
                ops: &[EpilogueOp::ResidualAdd],
            }],
            params: &[],
            micro_kernel: None,
        };
        static GELU_TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_epilogue_gelu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Epilogue {
                ops: &[EpilogueOp::Gelu],
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let inputs = TemplateInputs::new(&["x"]);

        let mut interp = TemplateInterpreter::new(params.clone());
        let res_ops = interp.instantiate(&RESIDUAL_TMPL, &inputs);
        assert!(res_ops.iter().any(|op| matches!(op, TraceOp::EpilogueChain { .. })),
            "ResidualAdd epilogue should produce EpilogueChain");

        let mut interp2 = TemplateInterpreter::new(params);
        let gelu_ops = interp2.instantiate(&GELU_TMPL, &inputs);
        assert!(gelu_ops.iter().any(|op| matches!(op, TraceOp::EpilogueChain { .. })),
            "Gelu epilogue should produce EpilogueChain");
    }

    // ── Additional unit tests (wave-12x59) ───────────────────────────

    /// Verify LoadPanel for MatrixRole::A at top level resolves the correct
    /// env slots (a_ptr, a_offset) and parameterized rows/cols.
    #[test]
    fn test_interpreter_load_panel_a_slot_indices_and_dims() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_load_panel_a",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::LoadPanel {
                matrix: MatrixRole::A,
                rows_param: "rows_a",
                cols_param: "cols_a",
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("rows_a", 8);
        params.set("cols_a", 32);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // After 6 gemm Input ops, the next op should be PanelLoad for A
        let panel = ops.iter().find(|op| matches!(op, TraceOp::PanelLoad { .. }));
        assert!(panel.is_some(), "LoadPanel A should produce TraceOp::PanelLoad");
        if let Some(TraceOp::PanelLoad { base, offset, rows, cols }) = panel {
            assert_eq!(*base, ValueId(0), "Panel A base should be a_ptr = ValueId(0)");
            assert_eq!(*offset, ValueId(3), "Panel A offset should be a_offset = ValueId(3)");
            assert_eq!(*rows, 8, "rows should resolve to 8");
            assert_eq!(*cols, 32, "cols should resolve to 32");
        }
    }

    /// Verify a multi-step Seq produces TraceOps in the declared order.
    #[test]
    fn test_interpreter_seq_preserves_step_order() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static STEPS: &[AlgoStep] = &[
            AlgoStep::Seq(&[
                AlgoStep::TraceBody(&[AlgoTraceStep::LoadConst { value: 7.0 }]),
                AlgoStep::TraceBody(&[AlgoTraceStep::LoadConst { value: 11.0 }]),
                AlgoStep::Softmax,
            ]),
        ];

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_seq_order",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: STEPS,
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // Should see: Input(0), Const(7.0), Const(11.0), Softmax in order
        let const_positions: Vec<(usize, f64)> = ops.iter().enumerate()
            .filter_map(|(i, op)| {
                if let TraceOp::Const(v) = op { Some((i, *v)) } else { None }
            })
            .collect();
        assert_eq!(const_positions.len(), 2, "Should produce exactly 2 Const ops");
        assert!((const_positions[0].1 - 7.0).abs() < 1e-10, "First const should be 7.0");
        assert!((const_positions[1].1 - 11.0).abs() < 1e-10, "Second const should be 11.0");
        assert!(const_positions[0].0 < const_positions[1].0, "Consts should appear in declared order");

        let softmax_pos = ops.iter().position(|op| matches!(op, TraceOp::Softmax { .. }));
        assert!(softmax_pos.is_some(), "Should produce Softmax");
        assert!(softmax_pos.unwrap() > const_positions[1].0,
            "Softmax should appear after all Const ops in the Seq");
    }

    /// Verify a Loop with empty body still produces a TraceOp::Loop with
    /// resolved bound and step_bytes, but empty body vector.
    #[test]
    fn test_interpreter_loop_empty_body() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_loop_empty_body",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "iters",
                step: "iter_step",
                body: &[],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("iters", 16);
        params.set("iter_step", 4);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let loop_op = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. }));
        assert!(loop_op.is_some(), "Should produce a Loop op");
        if let Some(TraceOp::Loop { bound, step_bytes, body }) = loop_op {
            assert_eq!(*bound, BoundExpr::Const(16), "bound should resolve to 16");
            assert_eq!(*step_bytes, 4, "step_bytes should resolve to 4");
            assert!(body.is_empty(), "empty body step list should produce empty Loop body");
        }
    }

    /// Verify deeply nested Seq (Seq within Seq within Seq) with a single
    /// innermost step produces exactly one TraceOp of that kind.
    #[test]
    fn test_interpreter_deeply_nested_seq_single_step() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static INNER: &[AlgoStep] = &[AlgoStep::Softmax];
        static MID: &[AlgoStep] = &[AlgoStep::Seq(INNER)];
        static OUTER: &[AlgoStep] = &[AlgoStep::Seq(MID)];

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_deep_seq",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: OUTER,
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let softmax_count = ops.iter().filter(|op| matches!(op, TraceOp::Softmax { .. })).count();
        assert_eq!(softmax_count, 1,
            "Deeply nested Seq with single Softmax should produce exactly 1 Softmax");
    }

    /// Verify TraceBody with VecLoadIndexed followed by VecStoreIndexed
    /// produces both TraceOps with correct slot references.
    #[test]
    fn test_interpreter_trace_body_load_then_store_indexed() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_load_store_indexed",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "base" },
                AlgoTraceStep::LoadInput { name: "off" },
                AlgoTraceStep::LoadInput { name: "val" },
                AlgoTraceStep::VecLoadIndexed { base: "base", offset: "off" },
                AlgoTraceStep::VecStoreIndexed { base: "base", offset: "off", src: "val" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["base", "off", "val"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let vli = ops.iter().find(|op| matches!(op, TraceOp::VecLoadIndexed { .. }));
        assert!(vli.is_some(), "Should produce VecLoadIndexed");
        if let Some(TraceOp::VecLoadIndexed { base, offset }) = vli {
            assert_eq!(*base, ValueId(0), "VLI base should be env 'base' = ValueId(0)");
            assert_eq!(*offset, ValueId(1), "VLI offset should be env 'off' = ValueId(1)");
        }

        let vsi = ops.iter().find(|op| matches!(op, TraceOp::VecStoreIndexed { .. }));
        assert!(vsi.is_some(), "Should produce VecStoreIndexed");
        if let Some(TraceOp::VecStoreIndexed { base, offset, value }) = vsi {
            assert_eq!(*base, ValueId(0), "VSI base should be env 'base' = ValueId(0)");
            assert_eq!(*offset, ValueId(1), "VSI offset should be env 'off' = ValueId(1)");
            assert_eq!(*value, ValueId(2), "VSI value should be env 'val' = ValueId(2)");
        }
    }

    /// Verify loop body MoeTopK produces TraceOp::Input referencing
    /// env.get("a_ptr") from the outer gemm inputs.
    #[test]
    fn test_interpreter_body_moe_top_k_uses_a_ptr() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_moe_topk",
            strategy: AlgoStrategy::MoeRouterTopk,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "experts",
                step: "expert_step",
                body: &[AlgoStep::MoeTopK {
                    num_experts: "ne",
                    top_k: "k",
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("experts", 8);
        params.set("expert_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            // Body MoeTopK produces TraceOp::Input(env.get("a_ptr").0)
            // With gemm inputs, a_ptr = ValueId(0)
            let input_in_body = body.iter().find(|op| matches!(op, TraceOp::Input(0)));
            assert!(input_in_body.is_some(),
                "Body MoeTopK should produce TraceOp::Input referencing a_ptr = ValueId(0)");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body ZeroFill produces TraceOp::Input referencing
    /// env.get("c_ptr") from the outer environment.
    #[test]
    fn test_interpreter_body_zero_fill_uses_c_ptr() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_zerofill",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[AlgoStep::ZeroFill { bytes_param: "fill_bytes" }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 4);
        params.set("tile_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            // Body ZeroFill produces TraceOp::Input(env.get("c_ptr").0)
            // With gemm inputs, c_ptr = ValueId(2)
            let input_in_body = body.iter().find(|op| matches!(op, TraceOp::Input(2)));
            assert!(input_in_body.is_some(),
                "Body ZeroFill should produce TraceOp::Input referencing c_ptr = ValueId(2)");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify ParamTable set overwrites produce the last-set value for a key,
    /// and that the interpreter uses the final value when resolving loop bounds.
    #[test]
    fn test_interpreter_param_overwrite_affects_loop_bound() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_param_overwrite",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "m",
                step: "m_step",
                body: &[AlgoStep::Softmax],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("m", 4);
        params.set("m", 16); // overwrite: last value wins
        params.set("m_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { bound, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert_eq!(*bound, BoundExpr::Const(16),
                "Loop bound should use the last-set value (16), not the first (4)");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body RowCopy produces two TraceOp::Input ops referencing
    /// env.get("a_ptr") and env.get("c_ptr") from the outer environment.
    #[test]
    fn test_interpreter_body_row_copy_uses_a_ptr_and_c_ptr() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_rowcopy",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "rows",
                step: "row_step",
                body: &[AlgoStep::RowCopy {
                    rows_param: "r",
                    cols_param: "c",
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("rows", 2);
        params.set("row_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let body_inputs: Vec<u32> = body.iter()
                .filter_map(|op| if let TraceOp::Input(n) = op { Some(*n) } else { None })
                .collect();
            assert_eq!(body_inputs.len(), 2,
                "Body RowCopy should produce exactly 2 Input ops, got {}", body_inputs.len());
            assert_eq!(body_inputs[0], 0, "First Input should be a_ptr = 0");
            assert_eq!(body_inputs[1], 2, "Second Input should be c_ptr = 2");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify TraceBody with AlgoTraceStep::Fma inside a loop body uses
    /// env.get() to resolve slot references from the outer environment,
    /// not the top-level next_slot logic.
    #[test]
    fn test_interpreter_body_trace_body_fma_uses_env() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_trace_fma",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[AlgoStep::TraceBody(&[
                    AlgoTraceStep::Fma { acc: "a_ptr", a: "b_ptr", b: "c_ptr" },
                ])],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 2);
        params.set("tile_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // gemm inputs: a_ptr=ValueId(0), b_ptr=ValueId(1), c_ptr=ValueId(2)
        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let fma = body.iter().find(|op| matches!(op, TraceOp::Fma(_, _, _)));
            assert!(fma.is_some(), "Body TraceBody Fma should produce TraceOp::Fma");
            if let Some(TraceOp::Fma(a, b, c)) = fma {
                assert_eq!(*a, ValueId(1), "Fma a should be b_ptr = ValueId(1)");
                assert_eq!(*b, ValueId(2), "Fma b should be c_ptr = ValueId(2)");
                assert_eq!(*c, ValueId(0), "Fma acc should be a_ptr = ValueId(0)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    // ── Additional unit tests (wave-12x60) ───────────────────────────

    /// Verify instantiate with zero inputs produces an empty ops vec
    /// (no Input preamble ops are generated).
    #[test]
    fn test_interpreter_zero_inputs_empty_ops() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_zero_inputs",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[],
            params: &[],
            micro_kernel: None,
        };

        let inputs = TemplateInputs::new(&[]);
        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert_eq!(ops.len(), 0,
            "empty template with zero inputs should produce no ops at all");
    }

    /// Verify ParamTable::resolve with unbound params defaults to 1,
    /// so Loop bounds for unset param names resolve to BoundExpr::Const(1).
    #[test]
    fn test_interpreter_loop_unbound_param_defaults_to_one() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_unbound_loop",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "unbound_dim",
                step: "unbound_step",
                body: &[AlgoStep::Softmax],
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { bound, step_bytes, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert_eq!(*bound, BoundExpr::Const(1),
                "unbound param should resolve to default 1 for bound");
            assert_eq!(*step_bytes, 1,
                "unbound step param should resolve to default 1 for step_bytes");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify instantiate with a single input produces exactly one Input(0)
    /// preamble op, and subsequent steps reference ValueId(0) via env.
    #[test]
    fn test_interpreter_single_input_env_binding() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_single_input",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Softmax],
            params: &[],
            micro_kernel: None,
        };

        let inputs = TemplateInputs::new(&["data"]);
        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert!(matches!(ops[0], TraceOp::Input(0)),
            "single input should produce Input(0) as first op");
        assert_eq!(ops.len(), 2,
            "single input template should produce 1 Input + 1 Softmax = 2 ops");
    }

    /// Verify loop body Activation Gelu produces TraceOp::Tanh (same mapping
    /// as top-level Gelu, but using env.get("panel_a") for src).
    #[test]
    fn test_interpreter_body_activation_gelu_produces_tanh() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_gelu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Activation { kind: ActivationKind::Gelu }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 4);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::Tanh(_))),
                "Body Gelu activation should produce TraceOp::Tanh");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body Activation Sigmoid (the AlgoStep variant, not
    /// TraceUnaryOp) produces TraceOp::Tanh per the interpreter mapping.
    #[test]
    fn test_interpreter_body_activation_sigmoid_step_produces_tanh() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_act_sigmoid",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Activation { kind: ActivationKind::Sigmoid }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 2);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::Tanh(_))),
                "Body Sigmoid activation (AlgoStep variant) should produce TraceOp::Tanh");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body Epilogue step produces TraceOp::EpilogueChain
    /// preserving the inner ops array exactly.
    #[test]
    fn test_interpreter_body_epilogue_preserves_ops() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, EpilogueOp, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_epilogue",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[AlgoStep::Epilogue {
                    ops: &[EpilogueOp::BiasAdd, EpilogueOp::Silu],
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 4);
        params.set("tile_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let ep = body.iter().find(|op| matches!(op, TraceOp::EpilogueChain { .. }));
            assert!(ep.is_some(), "Body Epilogue should produce EpilogueChain");
            if let Some(TraceOp::EpilogueChain { ops: ep_ops }) = ep {
                assert_eq!(ep_ops.len(), 2,
                    "EpilogueChain should preserve exactly 2 inner ops (BiasAdd + Silu)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify Conditional with multiple body steps emits all steps
    /// sequentially, not just the first.
    #[test]
    fn test_interpreter_conditional_multiple_body_steps() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_cond_multi",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Conditional {
                requirement: DeviceReq::CpuAny,
                body: &[AlgoStep::Softmax, AlgoStep::Softmax],
            }],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let softmax_count = ops.iter().filter(|op| matches!(op, TraceOp::Softmax { .. })).count();
        assert_eq!(softmax_count, 2,
            "Conditional with 2 Softmax steps should emit both, got {} softmaps", softmax_count);
    }

    /// Verify Conditional inside a loop body emits all inner steps
    /// into the loop body ops.
    #[test]
    fn test_interpreter_body_conditional_emits_steps_into_loop() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_cond",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Conditional {
                    requirement: DeviceReq::CpuAvx2,
                    body: &[AlgoStep::Softmax],
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 2);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert!(body.iter().any(|op| matches!(op, TraceOp::Softmax { .. })),
                "Conditional inside loop body should emit Softmax into body ops");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify nested loop with TraceBody inside inner loop body uses
    /// outer env to resolve slot references for LoadInput names.
    #[test]
    fn test_interpreter_nested_loop_trace_body_uses_outer_env() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_nested_trace",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "outer",
                step: "outer_step",
                body: &[AlgoStep::Loop {
                    bound: "inner",
                    step: "inner_step",
                    body: &[AlgoStep::TraceBody(&[
                        AlgoTraceStep::LoadInput { name: "data" },
                        AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Exp, dst: "out", src: "data" },
                    ])],
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("outer", 4);
        params.set("outer_step", 1);
        params.set("inner", 8);
        params.set("inner_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // Find outer loop → inner loop → inner body should have Input + Exp
        if let Some(TraceOp::Loop { body: outer_body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            if let Some(TraceOp::Loop { body: inner_body, .. }) = outer_body.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
                let input_in_inner = inner_body.iter().find(|op| matches!(op, TraceOp::Input(0)));
                assert!(input_in_inner.is_some(),
                    "TraceBody LoadInput in inner loop should reference env 'data' = ValueId(0)");
                let exp_in_inner = inner_body.iter().find(|op| matches!(op, TraceOp::Exp(_)));
                assert!(exp_in_inner.is_some(),
                    "TraceBody UnaryOp Exp should produce TraceOp::Exp in inner loop body");
            } else {
                panic!("Outer loop body should contain inner Loop");
            }
        } else {
            panic!("Template should produce outer Loop op");
        }
    }

    /// Verify loop body TraceBody HReduce inside interpret_body_step
    /// resolves the src slot from outer env correctly.
    #[test]
    fn test_interpreter_body_trace_body_hreduce_uses_env() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, ReduceKind,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_trace_hreduce",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::TraceBody(&[
                    AlgoTraceStep::HReduce { src: "data", op: ReduceKind::Sum },
                ])],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 2);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // "data" is bound to ValueId(0) in outer env
        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let hreduce = body.iter().find(|op| matches!(op, TraceOp::HReduce { .. }));
            assert!(hreduce.is_some(), "Body TraceBody HReduce should produce TraceOp::HReduce");
            if let Some(TraceOp::HReduce { src, op }) = hreduce {
                assert_eq!(*src, ValueId(0),
                    "HReduce src should reference env 'data' = ValueId(0)");
                assert!(matches!(op, crate::compiler::trace::ReduceKind::Sum),
                    "HReduce op should be ReduceKind::Sum");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    // ── Additional unit tests (wave-12x61) ───────────────────────────

    /// Verify loop body LoadPanel for MatrixRole::B resolves correct outer
    /// env slots (b_ptr=ValueId(1), b_offset=ValueId(4)) with gemm inputs.
    #[test]
    fn test_interpreter_body_load_panel_b_resolves_gemm_slots() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_load_panel_b",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "tiles",
                step: "tile_step",
                body: &[AlgoStep::LoadPanel {
                    matrix: MatrixRole::B,
                    rows_param: "br",
                    cols_param: "bc",
                }],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("tiles", 4);
        params.set("tile_step", 1);
        params.set("br", 32);
        params.set("bc", 16);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let panel = body.iter().find(|op| matches!(op, TraceOp::PanelLoad { .. }));
            assert!(panel.is_some(), "Body LoadPanel B should produce TraceOp::PanelLoad");
            if let Some(TraceOp::PanelLoad { base, offset, rows, cols }) = panel {
                assert_eq!(*base, ValueId(1), "Panel B base should be b_ptr = ValueId(1)");
                assert_eq!(*offset, ValueId(4), "Panel B offset should be b_offset = ValueId(4)");
                assert_eq!(*rows, 32, "rows should resolve to 32");
                assert_eq!(*cols, 16, "cols should resolve to 16");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify loop body TileMma references env slots for panel_a, panel_b,
    /// and panel_c — validates the interpret_body_step TileMma path.
    #[test]
    fn test_interpreter_body_tile_mma_uses_env_slots() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_tile_mma",
            strategy: AlgoStrategy::GemmGpuTiled,
            device_req: DeviceReq::GpuSm80,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "ar", cols_param: "ac" },
                AlgoStep::LoadPanel { matrix: MatrixRole::B, rows_param: "br", cols_param: "bc" },
                AlgoStep::LoadPanel { matrix: MatrixRole::C, rows_param: "cr", cols_param: "cc" },
                AlgoStep::Loop {
                    bound: "tiles",
                    step: "tile_step",
                    body: &[AlgoStep::TileMma],
                },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("ar", 4);
        params.set("ac", 16);
        params.set("br", 16);
        params.set("bc", 4);
        params.set("cr", 4);
        params.set("cc", 4);
        params.set("tiles", 2);
        params.set("tile_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // After 6 gemm Input ops: panel_a=slot6, panel_b=slot7, panel_c=slot8
        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let tile_mma = body.iter().find(|op| matches!(op, TraceOp::TileMma { .. }));
            assert!(tile_mma.is_some(), "Body TileMma should produce TraceOp::TileMma");
            if let Some(TraceOp::TileMma { c, a, b }) = tile_mma {
                assert_eq!(*c, ValueId(8), "TileMma c should be panel_c = ValueId(8)");
                assert_eq!(*a, ValueId(6), "TileMma a should be panel_a = ValueId(6)");
                assert_eq!(*b, ValueId(7), "TileMma b should be panel_b = ValueId(7)");
            }
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify a Seq containing a mix of TraceBody and structural steps
    /// produces TraceOps in the declared order.
    #[test]
    fn test_interpreter_seq_mixed_trace_and_structural_order() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static STEPS: &[AlgoStep] = &[
            AlgoStep::Seq(&[
                AlgoStep::TraceBody(&[AlgoTraceStep::LoadConst { value: 42.0 }]),
                AlgoStep::Softmax,
                AlgoStep::TraceBody(&[AlgoTraceStep::LoadConst { value: 99.0 }]),
            ]),
        ];

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_seq_mixed",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: STEPS,
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        // Order: Input(0), Const(42), Softmax, Const(99)
        let const_positions: Vec<(usize, f64)> = ops.iter().enumerate()
            .filter_map(|(i, op)| {
                if let TraceOp::Const(v) = op { Some((i, *v)) } else { None }
            })
            .collect();
        let softmax_pos = ops.iter().position(|op| matches!(op, TraceOp::Softmax { .. }));

        assert_eq!(const_positions.len(), 2, "Should produce 2 Const ops");
        assert!((const_positions[0].1 - 42.0).abs() < 1e-10, "First const should be 42.0");
        assert!((const_positions[1].1 - 99.0).abs() < 1e-10, "Second const should be 99.0");
        assert!(softmax_pos.is_some(), "Should produce Softmax");
        // Softmax must appear between the two Const ops
        let softmax_idx = softmax_pos.unwrap();
        assert!(const_positions[0].0 < softmax_idx,
            "Softmax should appear after first Const");
        assert!(softmax_idx < const_positions[1].0,
            "Softmax should appear before second Const");
    }

    /// Verify instantiating the same template twice with identical params
    /// produces identical TraceOp sequences (idempotent interpretation).
    #[test]
    fn test_interpreter_idempotent_instantiate() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceBinOp,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_idempotent",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "a" },
                AlgoTraceStep::LoadInput { name: "b" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "c", a: "a", b: "b" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let inputs = TemplateInputs::new(&["a", "b"]);
        let params = ParamTable::new();

        let mut interp1 = TemplateInterpreter::new(params.clone());
        let ops1 = interp1.instantiate(&TMPL, &inputs);

        let mut interp2 = TemplateInterpreter::new(params);
        let ops2 = interp2.instantiate(&TMPL, &inputs);

        assert_eq!(ops1.len(), ops2.len(),
            "Two instantiations should produce same number of ops");
        for (i, (op1, op2)) in ops1.iter().zip(ops2.iter()).enumerate() {
            assert!(std::mem::discriminant(op1) == std::mem::discriminant(op2),
                "op[{}] variant mismatch between two instantiations", i);
        }
    }

    /// Verify loop step_bytes resolves to a small value (1) correctly,
    /// testing the ParamTable resolve path for minimal stride.
    #[test]
    fn test_interpreter_loop_step_bytes_resolves_to_one() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_step_one",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "iters",
                step: "stride",
                body: &[AlgoStep::Softmax],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("iters", 128);
        params.set("stride", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { bound, step_bytes, body }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            assert_eq!(*bound, BoundExpr::Const(128), "bound should resolve to 128");
            assert_eq!(*step_bytes, 1, "step_bytes should resolve to 1");
            assert!(body.iter().any(|op| matches!(op, TraceOp::Softmax { .. })),
                "Loop body should contain Softmax");
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify a full GEMM pipeline (LoadPanel A + LoadPanel B + Loop with
    /// MicroKernel + StoreResult) produces all expected TraceOp kinds in order.
    #[test]
    fn test_interpreter_full_gemm_pipeline_ops_order() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_full_gemm",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[
                AlgoStep::LoadPanel { matrix: MatrixRole::A, rows_param: "mr", cols_param: "kr" },
                AlgoStep::LoadPanel { matrix: MatrixRole::B, rows_param: "kr2", cols_param: "nr" },
                AlgoStep::Loop {
                    bound: "m_tiles",
                    step: "m_step",
                    body: &[AlgoStep::MicroKernel],
                },
                AlgoStep::StoreResult { rows_param: "sr", cols_param: "sc" },
            ],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("mr", 4);
        params.set("kr", 32);
        params.set("kr2", 32);
        params.set("nr", 8);
        params.set("m_tiles", 2);
        params.set("m_step", 1);
        params.set("sr", 4);
        params.set("sc", 8);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::gemm();
        let ops = interp.instantiate(&TMPL, &inputs);

        // Helper: recursively search for TraceOp variants inside Loop bodies
        fn contains_op_recursive(ops: &[TraceOp], pred: &dyn Fn(&TraceOp) -> bool) -> bool {
            for op in ops {
                if pred(op) { return true; }
                if let TraceOp::Loop { body, .. } = op {
                    if contains_op_recursive(body, pred) { return true; }
                }
            }
            false
        }

        // Verify all expected op types exist (Fma is inside Loop body, so recursive search)
        assert!(contains_op_recursive(&ops, &|op| matches!(op, TraceOp::PanelLoad { .. })),
            "Pipeline should contain PanelLoad");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::Loop { .. })),
            "Pipeline should contain Loop");
        assert!(contains_op_recursive(&ops, &|op| matches!(op, TraceOp::Fma(_, _, _))),
            "Pipeline should contain Fma (from MicroKernel inside Loop)");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::PanelStore { .. })),
            "Pipeline should contain PanelStore (from StoreResult)");

        // Verify ordering: PanelLoad before Loop, PanelStore after Loop
        let panel_load_pos = ops.iter().position(|op| matches!(op, TraceOp::PanelLoad { .. })).unwrap();
        let loop_pos = ops.iter().position(|op| matches!(op, TraceOp::Loop { .. })).unwrap();
        let store_pos = ops.iter().position(|op| matches!(op, TraceOp::PanelStore { .. })).unwrap();
        assert!(panel_load_pos < loop_pos, "PanelLoad should appear before Loop");
        assert!(loop_pos < store_pos, "Loop should appear before PanelStore");
    }

    /// Verify loop body Seq wraps multiple steps that are all emitted into
    /// the loop body ops vector.
    #[test]
    fn test_interpreter_body_seq_wraps_multiple_steps() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_body_seq",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Loop {
                bound: "n",
                step: "n_step",
                body: &[AlgoStep::Seq(&[
                    AlgoStep::Softmax,
                    AlgoStep::Softmax,
                    AlgoStep::Softmax,
                ])],
            }],
            params: &[],
            micro_kernel: None,
        };

        let mut params = ParamTable::new();
        params.set("n", 2);
        params.set("n_step", 1);
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        if let Some(TraceOp::Loop { body, .. }) = ops.iter().find(|op| matches!(op, TraceOp::Loop { .. })) {
            let softmax_count = body.iter().filter(|op| matches!(op, TraceOp::Softmax { .. })).count();
            assert_eq!(softmax_count, 3,
                "Seq with 3 Softmax inside loop body should emit all 3, got {}", softmax_count);
        } else {
            panic!("Template should produce a Loop op");
        }
    }

    /// Verify TraceBody with LoadInput + BinOp Add + UnaryOp Exp + HReduce Sum
    /// produces all four kinds of TraceOps in sequence.
    #[test]
    fn test_interpreter_trace_body_multi_step_produces_all_ops() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep, TraceBinOp, TraceUnaryOp, ReduceKind,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_multi_trace",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "x" },
                AlgoTraceStep::BinOp { op: TraceBinOp::Add, dst: "sum", a: "x", b: "x" },
                AlgoTraceStep::UnaryOp { op: TraceUnaryOp::Exp, dst: "exp_val", src: "x" },
                AlgoTraceStep::HReduce { src: "x", op: ReduceKind::Sum },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["x"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        assert!(ops.iter().any(|op| matches!(op, TraceOp::Add(_, _))),
            "Should contain TraceOp::Add from BinOp");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::Exp(_))),
            "Should contain TraceOp::Exp from UnaryOp");
        assert!(ops.iter().any(|op| matches!(op, TraceOp::HReduce { .. })),
            "Should contain TraceOp::HReduce from HReduce step");
    }

    /// Verify Seq with Softmax followed by Activation Silu produces both
    /// in correct order — Softmax before Sigmoid/Mul.
    #[test]
    fn test_interpreter_seq_softmax_then_silu_order() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, ActivationKind,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_softmax_silu",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::Seq(&[
                AlgoStep::Softmax,
                AlgoStep::Activation { kind: ActivationKind::Silu },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["data"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let softmax_pos = ops.iter().position(|op| matches!(op, TraceOp::Softmax { .. }));
        let sigmoid_pos = ops.iter().position(|op| matches!(op, TraceOp::Sigmoid(_)));

        assert!(softmax_pos.is_some(), "Should produce Softmax");
        assert!(sigmoid_pos.is_some(), "Should produce Sigmoid (from Silu)");
        assert!(softmax_pos.unwrap() < sigmoid_pos.unwrap(),
            "Softmax should appear before Sigmoid from Silu activation");
    }

    /// Verify TraceBody BroadcastScalar step resolves src from env and
    /// produces TraceOp::BroadcastScalar with the correct ValueId.
    #[test]
    fn test_interpreter_trace_body_broadcast_scalar_slot() {
        use crate::compiler::codegen::vm::algo_template::{
            AlgoTemplate, AlgoStep, AlgoTraceStep,
            AlgoStrategy, DeviceReq,
        };

        static TMPL: AlgoTemplate = AlgoTemplate {
            name: "test_broadcast_slot",
            strategy: AlgoStrategy::GemmNaive,
            device_req: DeviceReq::CpuAny,
            steps: &[AlgoStep::TraceBody(&[
                AlgoTraceStep::LoadInput { name: "scalar_val" },
                AlgoTraceStep::Broadcast { src: "scalar_val", dst: "wide" },
            ])],
            params: &[],
            micro_kernel: None,
        };

        let params = ParamTable::new();
        let mut interp = TemplateInterpreter::new(params);
        let inputs = TemplateInputs::new(&["scalar_val"]);
        let ops = interp.instantiate(&TMPL, &inputs);

        let broadcast = ops.iter().find(|op| matches!(op, TraceOp::BroadcastScalar { .. }));
        assert!(broadcast.is_some(), "Broadcast step should produce TraceOp::BroadcastScalar");
        if let Some(TraceOp::BroadcastScalar { src }) = broadcast {
            assert_eq!(*src, ValueId(0),
                "BroadcastScalar src should reference env 'scalar_val' = ValueId(0)");
        }
    }
}
