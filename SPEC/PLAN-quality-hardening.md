# 企业级质量加固方案

> 目标：消除代码库中所有 placeholder/stub/hardcoded fallback/标量循环等低质量实现，达到企业级生产标准。

---

## 问题清单（8 类，按严重程度排序）

### P0: 静默错误（产生错误计算结果）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 1 | `x86_64.rs:3461-3467` | `emit_trace_ops_avx512` 中 `TraceOp::Log` 是 placeholder，只做 vmovups 复制输入 | AVX-512 独立寄存器分配路径的 Log 运算返回输入值而非 ln(x)，静默产生错误结果 |

**修复**：调用已有的 `emit_log_avx512(zmm30, zmm29, [zmm29, zmm31, acc])`，与其他两条 AVX-512 路径一致。

---

### P1: 静默降级（编译失败不报错）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 2 | `compiler/mod.rs:245-249` | `compile_layer()` 在 `#[cfg(not(feature = "jit-x86"))]` 时静默返回 `emit_stub_code()`（空函数） | 调用方拿到一个 no-op CompiledLayer，执行时不做任何计算，无任何警告 |
| 3 | `compiler/mod.rs:274-278` | `compile_graph()` 同上 | 同上 |
| 4 | `compiler/mod.rs:416-417` | 测试中 `emit_stub_code` 作为"编译成功"的验证 | 测试通过但实际没验证任何计算 |

**修复**：
- 非 JIT 路径返回 `Err(InferenceError::CompileError("JIT backend not enabled (feature jit-x86 required)".into()))` 而非静默 stub
- `emit_stub_code()` 标记为 `#[deprecated]`，仅保留给显式需要 no-op 的测试场景
- 测试中使用 stub 的地方加 `#[cfg(not(feature = "jit-x86"))]` 条件编译

---

### P2: 硬编码 fallback（绕过 registry 的降级路径）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 5 | `x86_64.rs:1177` | `emit_elementwise_chain` 在 registry 不可用时调用 `emit_elementwise_chain_hardcoded` | 硬编码只支持 SiLU/GELU/Add/Mul，其他算子静默跳过 |
| 6 | `x86_64.rs:1291-1336` | `emit_elementwise_chain_hardcoded` 的 tail 处理只做 copy 不做计算 | 尾部元素（elem_count % simd_width）结果错误 |

**修复**：
- `emit_elementwise_chain` 在 registry 为 None 时返回 `Err("ScalarOpRegistry required for elementwise chain codegen")`
- 删除 `emit_elementwise_chain_hardcoded` 和 `emit_chain_body`
- 所有调用路径确保传入 registry（`compile_layer` / `compile_graph` 已经构造 `ScalarOpRegistry::with_defaults()`）

---

### P3: 标量循环 fallback（性能问题）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 7 | `traits.rs:231-254` | `gemm_bias_act` 的 activation fallback 用标量 for 循环 | GELU 用硬编码常数 `0.7978845608` / `0.044715`，SiLU 用 `(-v).exp()`，无 SIMD |

**修复**：
- `gemm_bias_act` 默认实现改为调用已有的 SIMD 激活函数：`self.silu(c, c)` / `self.gelu(c, c)` / `self.relu(c, c)`
- 消除标量循环和硬编码常数

---

### P4: Kernels trait 的 unimplemented!() stub（~50 个方法）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 8 | `traits.rs:203-412` | 大量方法默认 body 是 `unimplemented!("xxx")`，运行时 panic | 生产环境调用未实现方法直接崩溃，无编译期保护 |

**修复**：按 SPEC REQ-OPS 状态分三类处理：

**A. 已有 SIMD 实现但 trait 默认 body 仍是 unimplemented（需要验证）**：
- 检查 `cpu_kernels/mod.rs` 中 `CpuKernels` 的 impl 是否覆盖了所有 🟢 状态的算子
- 如果已覆盖，trait 默认 body 不影响（因为 CpuKernels 覆写了）
- 但为防御性编程，将 `unimplemented!()` 改为 `panic!("Kernels::{op_name} not implemented for this backend")` 并附带更清晰的错误信息

**B. 量化 matmul dispatch 的 unimplemented（`cpu_kernels/mod.rs:1636,1652,1672`）**：
- 将 `unimplemented!("unsupported quant type")` 改为返回 `Result<(), KernelError>` 或在 dispatch 前做编译期 exhaustive match
- 当前 match 有 `_ => unimplemented!()` 兜底，应改为 exhaustive 枚举匹配

**C. 真正未实现的算子（SPEC 标记 🟡/🔴）**：
- 保留 `unimplemented!()` 但加 `#[doc(hidden)]` 和 `#[cold]` 标注
- 在 trait 文档中明确标注哪些方法是 required、哪些是 optional

---

### P5: emit_stub() 最小函数（设计层面）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 9 | `x86_64.rs:20` | `emit_stub()` 生成 push rbp; nops; pop rbp; ret | 本身是合理的最小可执行函数，但被 P1 中的路径滥用 |
| 10 | `aarch64.rs:27` | `emit_stub()` 生成单条 ret | 同上 |

**修复**：
- 保留 `emit_stub()` 但标记 `#[cfg(test)]`，仅用于测试 CompiledLayer 的 mmap/执行机制
- 生产路径不再调用

---

### P6: 测试中的 stub 依赖

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 11 | `compiler/mod.rs:415-443` | 多个测试用 `emit_stub_code` 验证"编译成功" | 测试不验证实际计算正确性 |
| 12 | `tests/compiler_e2e.rs:80-81,472` | E2E 测试用 stub | 同上 |

**修复**：
- 有 `jit-x86` feature 的测试改为验证真实 codegen 输出
- 无 JIT feature 的测试改为验证返回 `Err`

---

### P7: 注释中的 misleading 标注

| # | 位置 | 问题 |
|---|------|------|
| 13 | `compiler/mod.rs:416` | 注释 "Phase 3: Codegen (stub for now)" — "for now" 暗示临时状态 |
| 14 | `compiler/codegen/mod.rs:16` | "aarch64 backend is stub" — 实际已有实质实现 |

**修复**：更新注释反映当前实际状态。

---

## 修改文件清单

| 文件 | 修改类型 | 涉及问题 |
|------|---------|---------|
| `src/compiler/codegen/x86_64.rs` | 修复 Log placeholder + 删除 hardcoded fallback | #1, #5, #6 |
| `src/compiler/mod.rs` | 非 JIT 路径返回 Err + 更新注释 | #2, #3, #4, #13 |
| `src/compiler/codegen/emitter.rs` | 标记 emit_stub_code deprecated | #9, #10 |
| `src/traits.rs` | gemm_bias_act 改用 SIMD + unimplemented 改进 | #7, #8 |
| `src/cpu_kernels/mod.rs` | 量化 matmul dispatch exhaustive match | #8B |
| `src/compiler/codegen/mod.rs` | 更新注释 | #14 |
| `tests/compiler_e2e.rs` | 测试改为验证真实 codegen | #12 |

---

## 执行顺序

1. **#1** — AVX-512 Log placeholder（1 行修复，最高优先级）
2. **#5 #6** — 删除 hardcoded fallback + 强制 registry
3. **#2 #3** — 非 JIT 路径返回 Err
4. **#7** — gemm_bias_act 改用 SIMD 激活
5. **#8** — Kernels trait unimplemented 改进 + 量化 dispatch exhaustive match
6. **#9 #10 #11 #12** — emit_stub 限制 + 测试修复
7. **#13 #14** — 注释更新

## 验证

- `cargo test --features jit-x86` 全部通过
- `cargo test` (无 jit feature) 确认非 JIT 路径返回 Err
- 无 `grep -r "Placeholder\|stub for now\|hardcoded fallback"` 残留
