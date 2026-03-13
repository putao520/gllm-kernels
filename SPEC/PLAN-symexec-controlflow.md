# SymExec 复杂控制流支持 — 设计与实现

> 状态：Phase 1-5 全部完成 | 测试：83 pass | 代码：5638 行

## 一、现状（重构前）

### 1.1 架构

symexec 由三层组成：
- `decoder.rs` — 反汇编 `extern "C"` 函数，逐条喂给 engine
- `engine.rs` — `SymbolicExecutor`，维护寄存器→`SymValue` 映射，逐条更新符号状态
- `sym_value.rs` — 符号值 AST，支持代数化简

### 1.2 控制流缺陷（已修复）

`decoder.rs` 原始线性扫描的限制：
- 后向跳转（循环回边）→ 直接 `break`，只分析第一次迭代
- 前向跳转（条件分支）→ 直接 `continue`，只走 fall-through
- 多个顺序循环 → 第一个回边 break 后，后续循环完全不可见

### 1.3 算子 symexec 成功/失败矩阵（重构前）

| 算子 | 循环数 | SymExec | 失败原因 |
|------|--------|---------|---------|
| SiLU, GELU, ReLU | 1 | ✅ | 单循环 elementwise |
| SwiGLU, GeGLU | 1 | ✅ | 单循环 binary elementwise |
| VecAdd, VecMul | 1 | ✅ | 最简单情况 |
| Softmax | 3 | ❌ | 3-pass: max → exp-sum → normalize |
| RmsNorm | 2 | ❌ | 2-pass: sum_sq → normalize |
| LayerNorm | 3 | ❌ | 3-pass: mean → var → normalize |
| L2Normalize | 2 | ❌ | 2-pass: sum_sq → normalize |
| MeanPool | 2 | ❌ | 2-pass: sum → divide |
| GEMM, GemmBias | 3(嵌套) | ❌ | 三重嵌套循环 |
| RoPE | 2(嵌套) | ❌ | 双重嵌套循环 |
| Transpose | 2(嵌套) | ❌ | 双重嵌套循环 |
| MHA | 多层嵌套 | ❌ | 极复杂 |

20 个注册算子中，只有 7 个能 symexec 成功，其余 13 个 fallback 到手写 manual trace。

## 二、架构设计

### 2.1 核心思路

不做通用符号执行引擎，针对 scalar ops 的有限控制流模式做结构化分析：

1. 构建控制流图（CFG）
2. 识别自然循环结构（dominator tree + back-edge）
3. 对每个循环体独立做符号执行（复用现有 engine）
4. 将多个循环的 trace 组合为 `ComputePattern`
5. 对循环体内的条件分支做 diamond 合并
6. 对嵌套循环做深度分析（GEMM/RoPE/Transpose）

关键洞察：scalar ops 都是 `for i in 0..n` 的简单计数循环，循环之间通过标量累加器传递数据。

### 2.2 模块结构

```
src/compiler/symexec/
├── mod.rs              # pub mod cfg, loop_analyzer, branch_merger (cfg gated: jit-x86)
├── sym_value.rs        # 符号值 AST，含 Select 变体 + SelectKind
├── engine.rs           # SymbolicExecutor，含 snapshot/restore/xmm_state/stack_state
├── decoder.rs          # 反汇编 + build_cfg_from_fn + analyze_scalar_fn_structured
├── cfg.rs              # CFG 构建 + 自然循环检测 + LoopForest 嵌套关系
├── loop_analyzer.rs    # 循环体符号执行 + Reduction 检测 + 多 pass 组合 + 嵌套循环分析
└── branch_merger.rs    # 条件分支 diamond 检测 + 合并 + Select 节点生成
```

### 2.3 数据结构

#### CFG (`cfg.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchKind {
    Above, AboveEqual, Below, BelowEqual,
    Greater, GreaterEqual, Less, LessEqual,
    Equal, NotEqual,
    Sign, NotSign, Parity, NotParity,  // 罕见但需要处理
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Fallthrough(BlockId),
    Jump(BlockId),
    CondBranch { kind: BranchKind, taken: BlockId, fallthrough: BlockId },
    Return,
}

#[derive(Debug, Clone)]
pub struct DecodedInsn {
    pub mnemonic: String,
    pub operands: Vec<String>,
    pub addr: u64,
}

pub struct BasicBlock {
    pub id: BlockId,
    pub start_addr: u64,
    pub end_addr: u64,
    pub instructions: Vec<DecodedInsn>,
    pub terminator: Terminator,
}

pub struct ControlFlowGraph {
    pub blocks: BTreeMap<BlockId, BasicBlock>,
    pub entry: BlockId,
    pub successors: HashMap<BlockId, Vec<BlockId>>,
    pub predecessors: HashMap<BlockId, Vec<BlockId>>,
}

pub struct NaturalLoop {
    pub header: BlockId,
    pub body_blocks: HashSet<BlockId>,
    pub latch: BlockId,
    pub exits: Vec<BlockId>,   // 多出口支持
    pub ordinal: usize,
    pub depth: usize,
}

pub struct LoopForest {
    pub loops: Vec<NaturalLoop>,
    pub children: HashMap<BlockId, Vec<usize>>,
    pub top_level: Vec<usize>,
}
```

#### 循环分析 (`loop_analyzer.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionKind { Sum, Max, Min }

/// 累加器初始化方式
pub enum AccumulatorInit {
    Const(f64),              // xorps → 0.0, movss NEG_INF → -inf
    Symbolic(SymValue),      // 非常量初始化
}

pub struct LoopTrace {
    pub loop_header: BlockId,
    pub reductions: Vec<ReductionDetected>,
    pub unknown_mutations: Vec<(String, SymValue)>,
    pub body_block_count: usize,
}

pub struct ReductionDetected {
    pub register: String,
    pub kind: ReductionKind,
    pub init: AccumulatorInit,
    pub body_expr: SymValue,
}

pub struct MultiPassAnalysis {
    pub loop_traces: Vec<LoopTrace>,
    pub pattern: ComputePattern,
    pub num_loops: usize,
}
```

#### SymValue 扩展 (`sym_value.rs`)

```rust
pub enum SymValue {
    // ... 现有变体不变 ...
    Select {
        kind: SelectKind,
        cond_lhs: Box<SymValue>,
        cond_rhs: Box<SymValue>,
        true_val: Box<SymValue>,
        false_val: Box<SymValue>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectKind { Gt, Ge, Lt, Le, Eq, Ne }
```

simplify 规则：
- `Select(a > b, a, b)` → `Max(a, b)`（含 Ge）
- `Select(a < b, a, b)` → `Min(a, b)`（含 Le）
- 交换模式：`Select(a > b, b, a)` → `Min(a, b)`
- 常量条件折叠：`Select(Const op Const, t, f)` → `t` 或 `f`
- 相同分支折叠：`Select(cond, x, x)` → `x`

#### Executor 扩展 (`engine.rs`)

```rust
impl SymbolicExecutor {
    pub fn snapshot(&self) -> SymbolicExecutor;   // Clone-based deep copy
    pub fn restore(&mut self, snap: SymbolicExecutor);
    pub fn xmm_state(&self) -> HashMap<String, SymValue>;
    pub fn get_flags(&self) -> Option<(SymValue, SymValue)>;
    pub fn stack_state(&self) -> &HashMap<i64, SymValue>;
    pub fn set_stack(&mut self, offset: i64, val: SymValue);
}
```

linearize 中 Select 处理：
- `Gt`/`Ge` → `TraceOp::Max`
- `Lt`/`Le` → `TraceOp::Min`
- `Eq`/`Ne` → 回退到 true 分支

### 2.4 核心算法

#### CFG 构建

1. 线性扫描反汇编（iced-x86），收集所有跳转目标地址
2. 按跳转目标切分基本块
3. 构建 successor/predecessor 映射
4. 跳转到函数外部视为 `Return`

#### 自然循环检测

1. 计算 dominator tree（iterative dataflow）
2. 找 back-edges：edge (A→B) 是 back-edge 当且仅当 B dominates A
3. 对每条 back-edge 构建自然循环体（反向 BFS 从 latch 到 header）
4. 计算嵌套关系（superset check + tightest parent + BFS depth）

#### 循环体符号执行

1. snapshot executor 状态
2. 按地址排序 body blocks（近似拓扑序，对可约循环正确）
3. 调用 `execute_blocks_with_merging`（处理循环体内 diamond 分支）
4. 比较 pre/post XMM 状态，检测 reduction 模式

#### Reduction 检测

覆盖模式：
- `Add(pre, expr)` / `Add(expr, pre)` → Sum
- `Fma(a, b, pre)` → Sum of `a*b`
- `Fma(pre, 1.0, c)` → Sum of `c`
- `Max(pre, expr)` / `Max(expr, pre)` → Max
- `Min(pre, expr)` / `Min(expr, pre)` → Min
- 展开归约链：`Max(mem1, Max(mem2, acc))` → 递归识别

#### 多 Pass 组合

- Coalescing：连续无归约循环合并为单个逻辑 pass（向量化主循环 + 标量尾循环）
- 1 循环 + reduction → `Reduction`
- 1 循环 无 reduction → `Elementwise`
- 2 循环 (reduce → transform) → `NormLike` (RmsNorm, L2Normalize)
- 3 循环 (max → exp-sum → normalize) → `Softmax`
- 3 循环 (mean → var → normalize) → `NormLike` (LayerNorm)
- 超过 3 个逻辑循环 → 返回错误

#### 条件分支合并

1. `find_diamonds`：识别 diamond 结构（CondBranch → taken/fallthrough 各一个后继 → 汇合点相同）
2. `merge_diamond`：snapshot → 执行 taken → snapshot → 执行 fallthrough → 对每个 XMM/stack 差异创建 `Select` 节点
3. `execute_blocks_with_merging`：替代线性执行，自动检测并合并 diamond

#### 嵌套循环分析

1. 仅在 `max_depth >= 1` 且 `top_level.len() == 1` 时激活
2. 从最深层向浅层搜索有意义的循环 trace
3. 模式映射：
   - 3-deep + Sum(Mul/Fma) → `Gemm`
   - 2-deep + Sum(Mul/Fma) → `Gemm`（退化，M=1 或 N=1）
   - 2-deep 无归约 → `Injective`（RoPE, Transpose）
   - 3-deep 无 FMA → `Injective`（fallback）
4. 无有意义 trace 时使用 structural_fallback（3-deep → Gemm）

### 2.5 decoder.rs 接口

```rust
/// 构建 CFG（实际实现在 cfg.rs）
pub fn build_cfg_from_fn(fn_ptr: *const u8, sig: &ScalarFnSignature)
    -> Result<(ControlFlowGraph, PtrParamMap), SymExecError>;

/// 基于 CFG 做结构化符号执行
/// 嵌套分析优先于平坦分析，无循环时返回 Ok(None)
pub fn analyze_scalar_fn_structured(fn_ptr: *const u8, sig: &ScalarFnSignature)
    -> Result<Option<MultiPassAnalysis>, SymExecError>;

/// 向后兼容：原始线性扫描，保持不变
pub fn analyze_scalar_fn(fn_ptr: *const u8, sig: &ScalarFnSignature)
    -> Result<Vec<TraceOp>, SymExecError>;
```

### 2.6 三级 Fallback 集成（registry.rs）

`register_with_symexec_fallback` 实现三级 fallback 策略：

1. `auto_register_structured` — CFG 多循环结构化分析（最精确）
2. `auto_register` — 原始线性扫描 symexec
3. 手写 `ManualTrace` — 兜底

```rust
fn register_with_symexec_fallback(
    &mut self, key: OpKindKey, sig: ScalarFnSignature,
    default_op_kind: OpKind, manual_trace: ManualTrace,
) {
    // 1. 尝试结构化分析
    if let Ok(Some(_)) = self.auto_register_structured(...) { return; }
    // 2. 尝试线性扫描
    if self.auto_register(...).is_ok() { return; }
    // 3. 手写 trace 兜底
    self.register_manual(key, manual_trace);
}
```

所有 20 个算子（含 MeanPool）均通过此接口注册。

## 三、实现状态

### Phase 1: CFG 构建 + 循环检测 ✅

- `cfg.rs` (782 行)：`build_cfg_from_fn` + `find_loops` + `compute_dominators`
- 3 个测试

### Phase 2: 循环体符号执行 + Reduction 检测 ✅

- `loop_analyzer.rs` (1602 行)：`analyze_single_loop` + `detect_reduction_pattern`
- `engine.rs` 扩展：snapshot/restore/xmm_state/get_flags/stack_state/Clone
- 22 个测试

### Phase 3: 多 Pass 组合 → ComputePattern ✅

- `loop_analyzer.rs` 扩展：`combine_passes` + coalescing 逻辑
- `decoder.rs` 扩展：`analyze_scalar_fn_structured`
- 测试覆盖 RmsNorm, Softmax, LayerNorm 模式

### Phase 4: 条件分支合并 ✅

- `branch_merger.rs` (451 行)：`find_diamonds` + `merge_diamond` + `execute_blocks_with_merging`
- `sym_value.rs` 扩展：Select 变体 + SelectKind + simplify 规则
- `engine.rs` 扩展：linearize 中 Select → Max/Min
- 6 个测试

### Phase 5: 嵌套循环分析 ✅

- `loop_analyzer.rs` 扩展：`analyze_nested_loops` + structural_fallback
- `cfg.rs` 扩展：LoopForest 嵌套关系（superset check + depth BFS）
- 测试覆盖 GEMM, RoPE, Transpose 模式

## 四、测试矩阵

| 模块 | 测试数 | 覆盖内容 |
|------|--------|---------|
| cfg.rs | 3 | CFG 构建、循环检测、嵌套关系 |
| loop_analyzer.rs | 22 | Reduction 检测、多 pass 组合、嵌套循环、coalescing |
| branch_merger.rs | 6 | Diamond 检测、Select 生成、Max/Min 简化 |
| sym_value.rs | 10 | Select simplify、常量折叠、相同分支折叠 |
| engine.rs | 32 | 指令执行、snapshot/restore、linearize |
| decoder.rs | 11 | 端到端 structured analysis（含 MeanPool, RmsNorm, Softmax, LayerNorm, L2Normalize） |
| **总计** | **83** | |

## 五、已知限制

1. **不可约循环**: 按地址排序近似拓扑序，对不可约循环可能不正确（极罕见）
2. **嵌套 diamond**: 不处理 diamond 内嵌 diamond（标量循环体中极罕见）
3. **Eq/Ne Select**: linearize 中回退到 true 分支，false 分支成为 dead code
4. **eps 硬编码**: `combine_layer_norm` 中 eps 硬编码为 1e-5，依赖 codegen 覆盖实际值
5. **Display 比较**: 使用 `Display` 格式化比较符号值相等性，理论上可能有 false negatives

## 六、风险与回退

1. **CFG 构建失败**: 编译器优化可能产生非结构化控制流 → 回退到现有线性扫描
2. **Reduction 误检测**: 累加器初始化模式不匹配 → 回退到手写 trace
3. **向后兼容**: `analyze_scalar_fn` 保持原有签名，内部先尝试结构化分析，失败则 fallback
4. **增量验证**: 每个 Phase 完成后，所有现有测试必须继续通过
