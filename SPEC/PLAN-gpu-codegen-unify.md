# 统一 GPU Codegen 抽象层 — 设计与实现

> 状态：Phase 1-4 完成，Phase 5 待验证 | 测试：25 pass | 代码：gpu_ir/ 2472 行

## 一、现状（重构前）

### 1.1 CPU vs GPU 路径

- CPU: `TraceOp → algorithm.rs (emit_trace_op<E: SimdOps>) → SimdOps trait → x86_64/aarch64`
- GPU: `TraceOp → trace_op_to_ptx/hip/msl() → 字符串拼接`

### 1.2 重复量化

| 层次 | PTX | HIP | AIR |
|------|-----|-----|-----|
| TraceOp → 指令 | `trace_op_to_ptx` | `trace_op_to_hip` | `trace_op_to_msl` |
| Trace body 遍历 | `emit_trace_body_ptx` | `emit_trace_body_hip` | `emit_trace_body` |
| Elementwise kernel | `emit_elementwise_kernel_ptx` | `emit_elementwise_kernel_hip` | `emit_elementwise_kernel_from_trace` |
| NormLike kernel | `emit_normlike_kernel_ptx` | `emit_normlike_kernel_hip` | `emit_normlike_kernel` |
| Softmax kernel | `emit_softmax_kernel_ptx` | `emit_softmax_kernel_hip` | `emit_softmax_kernel` |
| emit_plan dispatch | L1740-1908 | L836-1015 | L650-812 |

每加一个新 TraceOp 变体，要在 4 处（含 CPU）各写一遍。

### 1.3 不可统一的部分

- GEMM: Tensor Core (wmma/mma.sync) vs MFMA vs simdgroup_matrix，指令级差异太大
- PTX 是汇编级（需要寄存器分配），HIP/MSL 是 C++ 级

## 二、架构设计

### 2.1 模块结构

```
src/compiler/codegen/
├── gpu_ir/
│   ├── mod.rs              # re-exports
│   ├── primitives.rs       # KernelParam + GpuCapabilities
│   ├── trace_emitter.rs    # GpuDialect trait + emit_trace_body<D> + 三后端 dialect impl
│   ├── kernel_builder.rs   # build_xxx_kernel<D> 统一骨架（6 个 builder）
│   └── plan_emitter.rs     # gpu_emit_plan<D> 统一 dispatch
├── ptx.rs                  # PtxCodeGen + standalone kernel 函数（MHA/RoPE 等）
├── ptx_gemm.rs             # PTX GEMM 子模块（wmma/mma.sync）
├── hip.rs                  # HipCodeGen + standalone kernel 函数
├── hip_gemm.rs             # HIP GEMM 子模块（scalar + MFMA）
├── air.rs                  # AirCodeGen + standalone kernel 函数
├── air_gemm.rs             # MSL GEMM 子模块（scalar + simdgroup_matrix）
└── ...                     # CPU 路径不变
```

### 2.2 GpuDialect Trait

```rust
pub trait GpuDialect {
    // ── 文件级 ──
    fn emit_header(&self, out: &mut String);

    // ── TraceOp 翻译 ──
    fn emit_trace_op(&self, out: &mut String, op: &TraceOp, idx: usize,
                     vars: &[String], tier: usize) -> String;

    // ── GPU 元数据 ──
    fn warp_size(&self) -> u32;
    fn capabilities(&self) -> GpuCapabilities;
    fn default_block_size(&self) -> u32 { 256 }  // 有默认实现

    // ── Kernel 结构 ──
    fn emit_kernel_start(&self, out: &mut String, name: &str, params: &[KernelParam],
                         shared_mem_bytes: usize);
    fn emit_kernel_end(&self, out: &mut String);

    // ── 线程索引 ──
    fn global_tid_expr(&self) -> &'static str;
    fn local_tid_expr(&self) -> &'static str;
    fn group_id_expr(&self) -> &'static str;

    // ── Shared Memory ──
    fn emit_shared_decl(&self, out: &mut String, name: &str, count: usize);
    fn emit_barrier(&self, out: &mut String);
    fn emit_shared_load(&self, out: &mut String, dst: &str, array: &str, index: &str);
    fn emit_shared_store(&self, out: &mut String, array: &str, index: &str, value: &str);

    // ── 控制流 ──
    fn emit_for_start(&self, out: &mut String, var: &str, start: &str,
                      limit: &str, stride: &str, label: &str);
    fn emit_for_end(&self, out: &mut String, var: &str, stride: &str, label: &str);
    fn emit_if_start(&self, out: &mut String, cond: &str, label: &str);
    fn emit_if_end(&self, out: &mut String, label: &str);
    fn emit_bounds_check_return(&self, out: &mut String, var: &str, limit: &str, label: &str);

    // ── 变量声明与赋值 ──
    fn emit_float_decl(&self, out: &mut String, name: &str, expr: &str);
    fn emit_float_assign(&self, out: &mut String, name: &str, expr: &str);

    // ── Global Memory ──
    fn emit_global_load(&self, out: &mut String, dst: &str, ptr: &str, index: &str);
    fn emit_global_store(&self, out: &mut String, ptr: &str, index: &str, value: &str);

    // ── Kernel 特化（各后端实现，委托到 standalone 函数）──
    fn emit_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]);
    fn emit_binary_elementwise_kernel(&self, out: &mut String, name: &str, body: &[TraceOp]);
    fn emit_gemm_kernel(&self, out: &mut String, name: &str, op_kind: &OpKind) -> Result<(), String>;
    fn emit_mha_kernel(&self, out: &mut String, name: &str, seq_len: usize,
                       num_heads: usize, head_dim: usize);
    fn emit_rope_kernel(&self, out: &mut String, name: &str, head_dim: usize, theta: f64);
    fn emit_reduction_kernel(&self, out: &mut String, name: &str,
                             identity: f64, combine: &[TraceOp]) -> Result<(), String>;
    fn emit_injective_kernel(&self, out: &mut String, name: &str, body: &[TraceOp],
                             num_inputs: usize, num_outputs: usize) -> Result<(), String>;
    fn emit_normlike_kernel(&self, out: &mut String, name: &str, reduce: &[TraceOp],
                            finalize: &[TraceOp], transform: &[TraceOp],
                            has_weight: bool, has_bias: bool, eps_override: Option<f64>);
    fn emit_softmax_kernel(&self, out: &mut String, name: &str);
    fn emit_meanpool_kernel(&self, out: &mut String, name: &str, seq_len: usize, hidden: usize);
    fn emit_dequantize_kernel(&self, out: &mut String, name: &str);
}
```

### 2.3 与原始 SPEC 的差异

| SPEC 设计 | 实际实现 | 原因 |
|-----------|---------|------|
| `emit_local_float` | `emit_float_decl` | 更精确的命名 |
| `emit_local_uint` | 不存在 | 未需要，uint 声明内联处理 |
| `emit_bounds_return` | `emit_bounds_check_return` | 更精确的命名 |
| `block_idx_expr` | `group_id_expr` | 跨后端统一术语（Metal 用 group） |
| `global_load_expr` → 返回 String | `emit_global_load` → void | PTX 需要多条指令，不适合返回表达式 |
| `emit_primitive(GpuPrimitive)` | 独立方法：`emit_barrier`, `emit_shared_decl/load/store` | 避免 match 开销，更直接 |
| `emit_injective_kernel` 有默认实现 | 无默认实现，三后端各自实现 | 所有后端必须显式处理 |
| 返回 `&'static str` | `global_tid_expr` 等返回 `&'static str` | 与 SPEC 一致 |

### 2.4 GpuPrimitive（已删除）

`GpuPrimitive` 枚举已从 `primitives.rs` 中移除。原始设计通过 `emit_primitive` 统一 dispatch，
实际实现改为 `GpuDialect` trait 上的独立方法（`emit_barrier`, `emit_shared_decl/load/store`）。
`primitives.rs` 现在仅包含 `KernelParam`、`ParamType`、`ParamQualifier`、`GpuCapabilities`。

### 2.5 辅助类型

```rust
pub struct KernelParam {
    pub name: String,
    pub ty: ParamType,
    pub qualifier: ParamQualifier,
}

pub enum ParamType { FloatPtr, Uint, Float }
pub enum ParamQualifier { Input, Output, Value }

pub struct GpuCapabilities {
    pub has_matrix_unit: bool,
    pub has_injective_codegen: bool,
}
```

### 2.6 统一 Trace Body

```rust
pub fn emit_trace_body<D: GpuDialect>(
    dialect: &D, out: &mut String, ops: &[TraceOp],
    tier: usize, input_bindings: &[String],
) -> String;
```

### 2.7 统一 Kernel Builder

`kernel_builder.rs` 中 6 个泛型 builder + 1 个辅助函数：

| 函数 | 状态 | 说明 |
|------|------|------|
| `build_elementwise_kernel<D>` | ✅ | unary elementwise |
| `build_binary_elementwise_kernel<D>` | ✅ | binary elementwise |
| `build_softmax_kernel<D>` | ✅ | 3-pass shared-memory reduction |
| `build_normlike_kernel<D>` | ✅ | 3-phase: reduce → finalize → transform，支持 weight/bias/eps |
| `build_meanpool_kernel<D>` | ✅ | mean pooling |
| `build_dequantize_kernel<D>` | ✅ | 量化解码 |
| `emit_tree_reduce<D>` | ✅ | 辅助：shared memory tree reduction |

未统一为泛型 builder 的 kernel（通过 trait 方法委托到后端 standalone 函数）：

| Kernel | 原因 |
|--------|------|
| Reduction | PTX/HIP/MSL 各自实现 `emit_reduction_kernel`，共享 memory tree-reduce 策略不同 |
| Injective | PTX/HIP/MSL 各自实现 `emit_injective_kernel`，参数绑定方式不同 |
| RoPE | 后端差异较大，委托到各后端 standalone 函数 |
| MHA | 后端差异较大，委托到各后端 standalone 函数 |
| GEMM | 指令级差异太大（wmma/MFMA/simdgroup_matrix），已提取到独立子模块 |

### 2.8 统一 emit_plan

```rust
pub fn gpu_emit_plan<D: GpuDialect>(
    dialect: &D, out: &mut String, plan: &FusionPlan,
    graph: &CompilerGraph, registry: Option<&ScalarOpRegistry>,
) -> Result<(), String>;
```

ComputePattern dispatch 覆盖：
- `Elementwise` → `build_elementwise_kernel<D>`
- `BinaryElementwise` → `build_binary_elementwise_kernel<D>`
- `NormLike` → `build_normlike_kernel<D>`
- `Reduction` → Softmax/MeanPool 特殊处理，其余走 `emit_reduction_kernel`
- `Gemm` → MHA 特殊处理，其余走 `emit_gemm_kernel`
- `Injective` → RoPE 特殊处理，其余走 `emit_injective_kernel`
- `QuantDecode` → `build_dequantize_kernel<D>`
- `Reshape`/`Transpose` → NOP skip

各后端 `MachineCodeEmitter::emit_plan` 变为 ~26 行 thin wrapper。

### 2.9 GEMM 子模块策略

GEMM kernel 已从各后端主文件提取到独立子模块，通过 `GpuDialect::emit_gemm_kernel` 调用：

| 后端 | 子模块 | 函数 |
|------|--------|------|
| PTX | `ptx_gemm.rs` (532 行) | `emit_gemm_kernel_ptx` |
| HIP | `hip_gemm.rs` (113 行) | `emit_gemm_kernel_hip`, `emit_gemm_mfma_kernel_hip` |
| MSL | `air_gemm.rs` (182 行) | `emit_gemm_kernel_msl`, `emit_gemm_simdgroup_msl`, `emit_gemm_bias_simdgroup_msl`, `emit_gemm_bias_kernel_msl` |

各子模块通过后端主文件 `pub(crate) use` 重新导出，外部调用路径不变。
提取原因：GEMM 指令级差异太大（wmma/mma.sync vs MFMA vs simdgroup_matrix），不适合统一为泛型 builder。

### 2.10 PTX 特殊处理

PTX 是汇编级语言，`emit_global_load` 等方法需要发射多条指令（地址计算 + load）。
`PtxDialect` 维护寄存器计数器（`rd_counter`, `r_counter`, `f_counter`）做简单线性分配。

### 2.11 后端 Dialect 实现

| 后端 | 结构体 | feature gate | 位置 |
|------|--------|-------------|------|
| PTX | `PtxDialect` | `jit-cuda` | `trace_emitter.rs` L254 |
| HIP | `HipDialect` | `jit-hip` | `trace_emitter.rs` L627 |
| MSL | `MslDialect` | `jit-metal` | `trace_emitter.rs` L951 |

三个 dialect 实现均在 `trace_emitter.rs` 中，通过 `impl GpuDialect for XxxDialect` 实现所有 trait 方法。
复杂 kernel（GEMM/MHA/RoPE 等）的 trait 方法委托到各后端文件（ptx.rs/hip.rs/air.rs）中的 standalone 函数。

## 三、实现状态

### Phase 1: GpuDialect trait + TraceOp 统一 ✅

- `trace_emitter.rs` (1400 行)：GpuDialect trait + emit_trace_body<D> + 三后端 dialect impl
- `primitives.rs` (46 行)：KernelParam, GpuCapabilities
- 7 个测试（含 legacy 等价性验证）

### Phase 2: Kernel 模板统一 ✅

- `kernel_builder.rs` (555 行)：6 个泛型 builder + emit_tree_reduce 辅助
- 10 个测试（HIP + MSL 覆盖）

### Phase 3: 统一 emit_plan dispatch ✅

- `plan_emitter.rs` (435 行)：gpu_emit_plan<D> 统一 dispatch + 8 个单元测试
- ptx.rs/hip.rs/air.rs 的 emit_plan 改为 ~26 行 thin wrapper

### Phase 4: GEMM 专用路径整理 ✅

- `ptx_gemm.rs` (532 行)：PTX GEMM 子模块，含 `emit_gemm_kernel_ptx`
- `hip_gemm.rs` (113 行)：HIP GEMM 子模块，含 `emit_gemm_kernel_hip` + `emit_gemm_mfma_kernel_hip`
- `air_gemm.rs` (182 行)：MSL GEMM 子模块，含 4 个 GEMM 函数
- 各后端主文件通过 `pub(crate) use` 重新导出，API 不变
- Reduction/Injective kernel 已在 PTX/MSL 补全实现（不再返回 `Err`）

### Phase 5: 新增算子端到端流程 ⏳ 待验证

重构完成后，新增算子的标准流程：
1. `scalar-ops/` 写 `extern "C" fn scalar_xxx(...)`
2. `registry.rs` 注册签名
3. symexec 自动提取 → ComputePattern
4. 如果是新的 ComputePattern → `kernel_builder.rs` 加一个 `build_xxx_kernel<D>`
5. 如果是已有 Pattern → 自动走通，零代码
6. 各后端的 `GpuDialect::emit_trace_op` 只在新增 TraceOp 变体时才需要修改

## 四、测试矩阵

| 模块 | 测试数 | 覆盖内容 |
|------|--------|---------|
| trace_emitter.rs | 7 | PTX/HIP/MSL dialect legacy 等价、TraceOp 覆盖、warp_size、空 body |
| kernel_builder.rs | 10 | elementwise/binary/softmax/normlike/meanpool/dequantize (HIP+MSL) |
| plan_emitter.rs | 8 | elementwise/binary/normlike/softmax/gemm/injective/reshape dispatch |
| **总计** | **25** | |

## 五、代码行数

| 模块 | 行数 | 说明 |
|------|------|------|
| gpu_ir/ | 2472 | trace_emitter 1408 + kernel_builder 555 + plan_emitter 435 + primitives 46 + mod 28 |
| ptx.rs | 1875 | MHA/RoPE standalone 函数（GEMM 已提取） |
| ptx_gemm.rs | 532 | PTX GEMM 子模块 |
| hip.rs | 1013 | MHA/RoPE standalone 函数（GEMM 已提取） |
| hip_gemm.rs | 113 | HIP GEMM 子模块 |
| air.rs | 1032 | MHA/RoPE standalone 函数（GEMM 已提取） |
| air_gemm.rs | 182 | MSL GEMM 子模块 |

GEMM 已提取到独立子模块，后端主文件瘦化。
emit_plan dispatch 已统一（~26 行/后端），plan_emitter 含 8 个单元测试。

## 六、已知问题

1. ~~**GpuPrimitive 死代码**~~: ✅ 已删除
2. ~~**plan_emitter 无测试**~~: ✅ 已补充 8 个单元测试
3. ~~**Reduction builder 缺失**~~: ✅ PTX/MSL/HIP 三后端均已实现 `emit_reduction_kernel`
4. **RoPE/MHA builder 缺失**: 未统一为泛型 builder，各后端独立实现
5. ~~**后端未充分瘦化**~~: ✅ GEMM 已提取到独立子模块（ptx_gemm.rs/hip_gemm.rs/air_gemm.rs）

## 七、风险与回退

1. **PTX 寄存器分配**: PTX 汇编级语言需要显式寄存器管理 → 用简单线性分配器
2. **输出等价性**: 重构可能改变变量命名/寄存器编号 → 用语义等价比较而非字符串比较
3. **向后兼容**: `MachineCodeEmitter` trait 签名不变，外部调用者无需修改
4. **增量迁移**: 旧函数标记 `#[deprecated]` 保留，parallel 测试通过后再删除
