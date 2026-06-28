# IR 不变量修复方案：副作用 op 的零输出合法化

> 触发问题：`diagnostics.rs::pre_check()` Check 1 (`op_has_output`) 对全体 op 统一要求 ≥1 输出，
> 错误拒绝了 StoreToken / CheckStopCondition 等纯副作用零输出 op。
> 状态：token 数据流建模已核实为 **模型 A**（固定内存 slot 副作用），置信度 95%。

---

## 1. 判定结论：模型 A（已核实，非待讨论）

| 层 | 证据 | 指向 |
|----|------|------|
| graph 构造 | `gllm/src/arch/auto_graph_fragments/build_graph.inc.rs:2381-2382` → `outputs: vec![]` | A |
| lowering | `gllm-kernels/src/compiler/codegen/vm/plan_lower/lower_op.inc.rs:374-425` → 纯 `VmInstr::StoreToken`/`CheckStopCondition` 写固定 buffer，零 VReg 返回 | A |
| 下一轮 token 来源 | 固定 slot `output_tokens_ptr`（栈参 offset 2），不走 SSA 边 | A |
| SPEC 40 | `output_tokens` 是 MegaKernelFn 外部 ABI 参数，非图内 tensor | A |
| registry | `defaults.inc.rs:1771-1801` → `num_outputs:1` | B（伪信号） |

**5 项证据中 4 项指向 A，唯一指向 B 的 registry `num_outputs:1` 是遗留债 + 占位 trace，未被实际 codegen 消费。**

故 token 不在 SSA 图中流动。StoreToken/CheckStopCondition 是生成循环的控制驱动（采样→写 token→检查停止），**零输出是正确语义**，Check 1 是错误的不变量。

---

## 2. 三层对齐方案

### Layer 1 — semantics.rs：引入显式副作用类别（SSOT）

拆分 `OpSemantics::Opaque`，新增 `SideEffect` 变体：

```rust
pub enum OpSemantics {
    Elementwise,
    Gemm,
    Reduction,
    Attention,
    Opaque,      // 有输出但优化器不可融合：Gather/Transpose/Reshape/MoEDispatchPacked...
    SideEffect,  // 纯副作用，零输出：StoreToken/CheckStopCondition/WriteLogits/EarlyExit/
                 //   SessionKvRestore/QTapSTG/GuardrailCheck/CotStepCheck/MmHiddenInject/MtpDraft
}
```

classify() 把 9 个纯副作用零输出 op 从 Opaque 迁到 SideEffect；33 个有输出的数据搬运 op 留在 Opaque。

**前置守卫**：grep `== OpSemantics::Opaque` 与 `OpSemantics::Opaque =>` 必须命中 0（当前融合代码只检查 `== Elementwise`，Opaque 是默认兜底桶 → 拆分行为安全）。

注意 `FusedRmsNormGemm` 当前误标 Opaque（实为 GEMM），顺手修正为 `Gemm`，不要混入 SideEffect。

### Layer 2 — diagnostics.rs：精确豁免

```rust
// Check 1: 非副作用 op 必须有 ≥1 输出；副作用 op 零输出合法
for op in &graph.ops {
    let is_side_effect = semantics::classify(&op.op) == OpSemantics::SideEffect;
    if op.outputs.is_empty() && !is_side_effect {
        errors.push(IrError::new("op_has_output", ...));
    }
    // 反向：副作用 op 若意外带输出，也应报错（精度强化，见 §3）
    if !op.outputs.is_empty() && is_side_effect {
        errors.push(IrError::new("side_effect_no_output", ...));
    }
}
```

豁免键 = `SideEffect`（精确 9 op），**不是** `Opaque`（44 op 超集）。后者会对 33 个有输出 op 放弃 has-output 检查能力。

### Layer 3 — registry defaults.inc.rs：消除矛盾

StoreToken/CheckStopCondition 的 `ComputePattern::Injective { num_outputs:1 }` 是把副作用 op 硬塞进计算 op 模子的占位 trace，且未被 codegen 消费（实走 `lower_op.inc.rs` 手写 lowering）。两选一：

- **(推荐) 改 `num_outputs: 0`** —— 与 graph `outputs:vec![]` 对齐，消除矛盾。
- (退路) 若 registry 不支持 num_outputs=0，则用 `ComputePattern::SideEffect` 专用变体表达"无计算语义"，彻底剥离 Injective 伪 trace。

---

## 3. 不变量强化（criterion 落 SPEC）

弱式 *"每个 op ≥1 输出"* → 双向强式：

> **IR-INV-SIDE-EFFECT**: `op.outputs.is_empty() ⟺ classify(op) == SideEffect`
> 即：当且仅当 op 属副作用集时零输出。非副作用 op 零输出 = bug（漏 lowering）；副作用 op 带输出 = bug（建模错）。

落点：`gllm-kernels/SPEC/26-VMINSTR-RATIONALIZATION.html` 新增 criterion，固化副作用 op 权威清单 + 双向不变量 + registry/graph num_outputs 对齐规则。

---

## 4. BCE 横扫（修复后强制，残留=0 才放行）

BUG 模式签名：**"IR/pass 对全体 op 统一假设了只对子类成立的属性（has output）"**。

横扫同类假设是否出现在：
- pre_check 其余 Check（已查：Check 2/3/4 对零输出 op 是空循环或一致，仅 Check 1 受影响）✓
- DCE / dead tensor 消除（零输出副作用 op 会不会被当死代码删除？必须验证副作用 op 不可消除）
- 调度 / scheduling pass（是否假设每 op 有输出 tensor 排序）
- verify def-before-use（已知一致：零输出 op 不产出 tensor，无 def 可被 use）
- auto_lower_trace 的输出消费处（registry num_outputs 改 0 后是否有路径仍读旧值）

---

## 5. 改动清单

| 文件 | 改动 |
|------|------|
| `gllm-kernels/src/compiler/semantics.rs` | 加 `OpSemantics::SideEffect`；9 副作用 op 迁类；修正 FusedRmsNormGemm→Gemm |
| `gllm-kernels/src/compiler/diagnostics.rs` | Check 1 双向化（豁免 SideEffect + 反向校验） |
| `gllm-kernels/src/compiler/registry_fragments/defaults.inc.rs` | StoreToken/CheckStopCondition num_outputs 1→0 |
| `gllm-kernels/SPEC/26-VMINSTR-RATIONALIZATION.html` | 新增 IR-INV-SIDE-EFFECT criterion |
| 融合代码（grep 守卫后确认） | 验证无 `== Opaque` 正向匹配 |
