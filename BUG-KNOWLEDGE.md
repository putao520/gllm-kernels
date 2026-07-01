# BUG-KNOWLEDGE.md — gllm-kernels BUG 模式知识库

> 每次 BCE 根治后沉淀，避免重复归因。按 patternId 倒序排列。
> gllm-kernels 仓专属（JIT codegen + VmInstr SSOT）。gllm 主仓另有独立 BUG-KNOWLEDGE.md。

## 根治总览

| 类别 | 条目数 | 根治 | 残留 | 备注 |
|------|--------|------|------|------|
| BCE-MIXED 算子级混合精度（BCE-20260630-MIXED） | 1 | 1 ✅ | 0 | emit_gemm_* 三段式 dtype（a/b/c）+ ctx.dtype per-op + accumulator_dtype() 标注 |
| BCE-OPTPASS 指令重写 dtype 丢失（BCE-20260630-OPTPASS） | 1 | 1 ✅ | 0 | substitute_loop_offset/forwarding match 绑定保留原 dtype，禁 `..` 丢弃 |

**全库残留总计**: 0

---

## BCE-MIXED — 算子级混合精度 dtype 感知

### smellClass: AP-HARDCODED-F32-EMIT（Pattern — emit_* 硬编码 F32 违背三段式语义）

**宪法依据**: ARCH-DTYPE-MIXED-PRECISION（CLAUDE.md 铁律 ARCH-DTYPE-JIT-TYPED）+ ARCH-DATA-FLOW-CONTRACT §11 emit dtype 传播契约 + GRAPH-SHAPE-DRIVEN-MEGA-KERNEL §0.8 dtype-sovereignty。每个 op 的**每个输入 tensor 都有独立 dtype**，必须各自从 TensorMeta 正向传播；同一个算子的 F32/BF16/混合精度实例是**三段不同的机器码**，由编译时 dtype 决定生成哪个。

**模式签名**: 算子 emit_* 函数在 load/accumulate/store 三段位置硬编码 `QuantPrecision::F32`，或 GEMM 接口虽收 dtype 参数但实现用 `let _ = (b_dtype, c_dtype);` 丢弃（"假完成"），违背 A-load=a_dtype / B-load=b_dtype / acc=accumulator_dtype / C-store=c_dtype 三段式语义。

```yaml
- patternId: BCE-20260630-MIXED
  title: emit_gemm_* 硬编码 F32 + GEMM 接口丢弃 b/c dtype（算子级混合精度断裂）
  layer: 范式缺陷
  smellClass: AP-HARDCODED-F32-EMIT
  codePattern:
    - "emit_gemm_blis_inline(... a_dtype, b_dtype, c_dtype, trans_b) 内部 let dtype = a_dtype; let _ = (b_dtype, c_dtype);  # 丢弃，假完成"
    - "VecLoad B 用 dtype: a_dtype  # 应为 b_dtype（权重独立 dtype）"
    - "Fma acc 用 dtype: a_dtype  # 应为 c_dtype.accumulator_dtype()（accumulate 位置 F32 合法但须显式标注）"
    - "ctx.dtype 全图统一 F32（graph_dtype 从 tensors.first() 推断），覆盖所有 op 的权重 load"
  triggerCondition:
    - 混合精度模型（A 激活 F32 + B 权重 BF16，或反之）的 GEMM / Attention / Norm 算子 emit
    - ctx.dtype 从 graph.tensors.first()（激活）推断后用于权重 load
    - GEMM blis/gpu 接口签名加了 a/b/c dtype 参数但实现体未真正使用
  detectionSignatures:
    structural: "CallExpression emit_gemm_* with dtype params followed by `let _ = (b_dtype, c_dtype);` or unused b_dtype/c_dtype"
    literal: "let dtype = a_dtype;"
    literal: "VecLoad { ..., dtype: a_dtype }  # B-matrix load 用激活 dtype"
    antipattern: "ctx.dtype 从 graph.tensors.first() 推断（激活 dtype 覆盖权重）"
  sameClassCriterion:
    - "任何 emit_* 的 B-matrix load dtype 必须独立从权重 tensor dtype 推断，禁止恒等于 a_dtype"
    - "accumulate 位置允许 F32（accumulator_dtype()），但必须显式标注，禁止隐式硬编码"
    - "ctx.dtype 必须 per-op（for_op）从 op.inputs[weight_idx].dtype 推断，禁止全图统一"
  fixTemplate:
    - "emit_gemm_blis_inline: a_elem=a_dtype.elem_bytes(); b_elem=b_dtype.elem_bytes(); c_elem=c_dtype.elem_bytes(); acc_dtype=c_dtype.accumulator_dtype(); VecLoad B 用 b_dtype; Fma 用 acc_dtype"
    - "ctx.dtype per-op 化：ctx.for_op(op) 从 op.inputs[weight_idx].dtype 推断，杠杆总闸撬动 RoPE/elementwise/softmax/attention"
    - "GemmOpLayout 加 a_dtype/b_dtype/c_dtype 三字段，OpImpl emit 据此多路传播"
  regressionAssertion:
    - "p05_dtype_matrix 测试：trans_b={true,false} × backend={CPU blis,GPU tiled} 4 组合，a_dtype=F32/b_dtype=BF16 混合精度，结构断言编译产物 VmInstr 中存在 VecLoad { dtype: BF16 }（B-load 用 b_dtype，非恒 a_dtype=F32）"
    - "反向回归：uniform F32 下不应出现异 dtype VecLoad（防 B-load 硬编码固定 dtype）"
  regressionTests:
    - "src/compiler/codegen/vm/e2e_tests_fragments/p05_dtype_matrix_tests.inc.rs（p05_dtype_matrix 模块，5 测试）"
    - "src/compiler/codegen/vm/gemm_impls.rs（BF16/F32 OpImpl 数值对齐，verify_op_impl_aligns_scalar，BF16 容差 1e-2 / F32 1e-5）"
  locations:
    - "src/compiler/codegen/vm/gemm_emit.rs（emit_gemm_blis_inline L205+ 三段式 dtype，B-load L294 用 b_dtype）"
    - "src/compiler/codegen/vm/gemm_emit.rs（emit_gemm_inline_with_epilogue L1139+）"
    - "src/compiler/codegen/vm/gemm_emit.rs（emit_gemm_gpu_tiled_inline / emit_gemm_gpu_pipelined）"
    - "src/compiler/codegen/vm/plan_lower/context.inc.rs（ctx.dtype per-op 化，P1 杠杆总闸）"
    - "src/compiler/codegen/vm/op_impl.rs（GemmOpLayout a_dtype/b_dtype/c_dtype 字段）"
  rootCause: "ctx.dtype 全图统一 F32（graph_dtype）+ GEMM blis/gpu 接口收 dtype 实现丢弃（假完成）+ emit_* load 位置硬编码 a_dtype 覆盖权重"
  fixCommitted:
    - "5d1e6cdb fix(BCE-20260630-MIXED-P0.5): GEMM blis/gpu 实现层真用 b/c dtype — 防假完成"
    - "8d419b9e fix(BCE-20260630-MIXED-P1): ctx.dtype per-op 化 — 杠杆总闸"
    - "e0777256 fix(BCE-20260630-MIXED-P5): Vision/Audio 算子 dtype 感知 — 三段式语义"
    - "6101258d fix(BCE-20260630-MIXED-test): 修 120 GEMM test debt + P0.5 dtype 矩阵测试"
  归因时间: 2026-06-30
  根治时间: 2026-06-30
  status: 根治 ✅ | residual: 0
```

---

## BCE-OPTPASS — opt_pass 指令重写 dtype 丢失

### smellClass: AP-DTYPE-DROP-IN-REWRITE（Pattern — 指令重写用 `..` 丢 dtype 后重建硬编码）

**宪法依据**: ARCH-DATA-FLOW-CONTRACT §11 emit dtype 传播契约（单向不可逆）+ ARCH-DTYPE-JIT-TYPED。opt_pass 对 VmInstr 的重写（循环展开 LoopOffset→Const、forwarding 等）必须**保留原指令的 dtype 字段**，禁止用 struct 重建 `..` 丢弃后重新硬编码 F32。

**模式签名**: opt_pass 的指令重写函数（substitute_loop_offset_in_instr / forwarding）用 `VmInstr::VecLoad { .., dtype: QuantPrecision::F32 }` 重建指令，`..` 丢弃原 dtype 后硬编码 F32。BF16/F16 weight load 经循环展开后 dtype 丢失为 F32 → 按错误字节宽度（4 vs 2）解码 → 数值错乱。

```yaml
- patternId: BCE-20260630-OPTPASS
  title: opt_pass substitute_loop_offset / forwarding 用 `..` 丢 dtype 重建硬编码 F32
  layer: 设计缺陷
  smellClass: AP-DTYPE-DROP-IN-REWRITE
  codePattern:
    - "VmInstr::VecLoad { offset: sub(oe), .., dtype: QuantPrecision::F32 }  # `..` 丢原 dtype，重建硬编码 F32"
    - "match instr { VmInstr::VecLoad { dst, base, offset, width, .. } => VmInstr::VecLoad { dst, base, offset: new, width, dtype: QuantPrecision::F32, predicate: None } }  # 丢弃 dtype + predicate"
  triggerCondition:
    - "opt_pass 循环展开（unroll_loop_body）调用 substitute_loop_offset_in_instr 替换 LoopOffset→Const"
    - "opt_pass 指令 forwarding / 重写路径"
    - "BF16/F16 weight load 指令经 opt_pass 重写"
  detectionSignatures:
    structural: "MatchExpression on VmInstr variant with `..` rest pattern, reconstruction with explicit `dtype: QuantPrecision::F32`"
    literal: "dtype: QuantPrecision::F32 在 opt_pass 指令重建 arm 内（非 accumulator 位置）"
    literal: "..  # 丢弃 dtype/predicate 后重建"
  sameClassCriterion:
    - "opt_pass 指令重写必须 match 绑定 dtype 字段（不用 `..` 丢弃），重建时透传原 dtype"
    - "accumulator 位置若需 F32，用 accumulator_dtype() 显式标注，禁止硬编码 QuantPrecision::F32"
    - "predicate 等其他字段同样必须 match 绑定透传"
  fixTemplate:
    - "match VmInstr::VecLoad { dst, base, offset, width, dtype, predicate } => VmInstr::VecLoad { dst, base, offset: sub(&offset), width, dtype, predicate }  # 全字段绑定透传"
    - "VecStore / Broadcast / Fma 同理：match 绑定 dtype 字段，重建时透传，禁 `..` + 硬编码 F32"
  regressionAssertion:
    - "BF16 VecLoad 经 substitute_loop_offset_in_instr 后 dtype 保持 BF16（不重置 F32）"
    - "F16 VecStore 经 substitute_loop_offset_in_instr 后 dtype 保持 F16"
    - "Broadcast dtype 透传（accumulator 位置若 F32 须显式 accumulator_dtype()）"
  regressionTests:
    - "src/compiler/codegen/vm/opt_pass.rs tests 模块（BCE-20260630-OPTPASS 段，TEST-OPTPASS-DTYPE-01/02/03）"
    - "@trace TEST-OPTPASS-DTYPE-01 [req:REQ-DTYPE-CHAIN-005] substitute_loop_offset_preserves_vecload_bf16_dtype"
    - "@trace TEST-OPTPASS-DTYPE-02 [req:REQ-DTYPE-CHAIN-005] substitute_loop_offset_preserves_vecstore_f16_dtype"
    - "@trace TEST-OPTPASS-DTYPE-03 [req:REQ-DTYPE-CHAIN-005] substitute_loop_offset_preserves_broadcast_dtype"
  locations:
    - "src/compiler/codegen/vm/opt_pass.rs（substitute_loop_offset_in_instr L312+，match 绑定 dtype 透传）"
    - "src/compiler/codegen/vm/opt_pass.rs（forwarding 路径 dtype 保留）"
  rootCause: "opt_pass 指令重写用 struct 重建 `..` 丢弃原 dtype 字段后硬编码 QuantPrecision::F32；BF16/F16 weight load 经循环展开后 dtype 丢失 → 按错误字节宽度解码"
  fixCommitted:
    - "bb616a48 fix(BCE-20260630-OPTPASS): opt_pass dtype 丢失根治 — substitute_loop_offset + forwarding 保留原 dtype"
  归因时间: 2026-06-30
  根治时间: 2026-06-30
  status: 根治 ✅ | residual: 0
```

---

## BCE-MEGA-KERNEL-EMIT-CTX-REFACTOR — 过程式 emit 长序列(B类, 待激活)

### smellClass: LONG-METHOD-PROCEDURAL-EMIT-SEQ（过程式 emit 长序列 + 游离编排状态，非 god-match dispatch）

**宪法依据**: P-2 复杂度限制(函数≤500行/圈复杂度≤10/参数≤5) + ARCH-JIT-GENERATOR 状态机架构(CodeGenerator{ctx,...}) + DEC-MKEMIT-001 A/B分层根治策略。

**分层根治策略(A类 vs B类，手法不可混用)**:
- **A类 — god-match dispatch**(lower_instr/auto_select/numerical_sim): 圈复杂度来自 match arm 数量。根治=瘦arm委托helper + 同语义arm查表自动化(auto_select 须遵 ARCH-AUTO-INSTR-SELECT 走 auto_lower_trace, 禁手写 TraceOp→VmInstr 映射) + category分组。`plan_lower/lower_op.inc.rs lower_op`(1577行)是**已根治范本**——瘦arm委托 lower_norm_v2/lower_*, arm多≠long_method, 不动。
- **B类 — 过程式长序列**(本条目 mega_kernel_emit): 圈复杂度来自线性emit序列+游离状态, match拆不动。根治=补编排层ctx + 阶段切method借用。

**模式签名(B类)**: 单函数内大量 `prog.emit(VmInstr)` 线性序列(197次/单函数, 占文件62%) + 大量游离编排级局部变量(25个), extract_function 抽出后 helper 需 15-25 参数 → 违反 P-2 并产生 long_parameter_list。入口本身已 long_parameter_list(15参数)。

```yaml
- patternId: BCE-MEGA-KERNEL-EMIT-CTX-REFACTOR
  title: compile_mega_kernel_vm 过程式 emit 长序列 + 入口 15 参数(B类长序列)
  layer: 范式缺陷
  smellClass: LONG-METHOD-PROCEDURAL-EMIT-SEQ
  decision: DEC-MKEMIT-001
  status: 待激活 ⏸ (P3 JIT codegen + aarch64 循环结构完成 + 回归基线稳定后激活)
  blockReason: "JIT codegen 正确性是 P0 铁律(NO_SILENT_FALLBACK/AUTO-INSTR-SELECT); 197 emit 序列重构期间寄存器/偏移/栈布局错误风险高且静默(产生错误结果无法常规测试发现); 须功能完整后做"
  codePattern:
    - "单函数内 197 次 prog.emit(VmInstr) 线性序列(占文件 316 次的 62%)"
    - "25 个游离编排级局部变量(scratchpad_batch/batch_ctx_ptr/vocab_bytes/width/topology/...)穿透 prologue/batch_mode/sampling 三阶段"
    - "入口函数 15 参数(plan/graph/alloc/registry/profile/hook/buffer_layout/bottleneck_map/virtual_activation/virtual_tensor_map/layout/debug_jit/.../resource_plan/topology)"
  triggerCondition:
    - "JIT emit 函数承载完整编排(prologue+batch_mode+sampling)而非委托分解"
    - "编排级状态以游离局部变量穿透多阶段, 无 ctx 收编"
  detectionSignatures:
    structural: "Function with >100 prog.emit() calls AND >15 local variables referenced across >2 logical phases"
    literal: "fn compile_mega_kernel_vm 单函数 prog.emit 计数 > 100"
    antipattern: "long_method + long_parameter_list 共现于 emit 入口"
  sameClassCriterion:
    - "JIT emit 入口函数 > 500 行且 prog.emit 密度高 + 游离编排状态 > 5 个 → 同类(B类)"
    - "区分 A类: 若长度来自 match arm 数量且 arm 已瘦委托 → 是 god-match(A类), 不归本条目"
  fixTemplate:
    - "补编排层 ctx(建议名 MegaKernelOrchestrator<'a>{ prog:&mut VmProgram, session:&CompileSession<'a>, abi:AbiPtrs, topology/sym_map/layout/vocab_size/width/... })"
    - "prologue/batch_mode/sampling 阶段切为 &mut self method, 零参数穿透"
    - "复用既有 CompileSession/LoweringContext(plan_lower/context.inc.rs L12/L47), 禁新建平行大杂烩 MegaKernelEmitCtx"
    - "入口只读编译上下文参数收进 CompileSession, 砍至 ≤6 参数"
  scope:
    - "仅 src/compiler/codegen/vm/mega_kernel_emit.rs"
    - "不波及 fusion_group_emit(1953行/13 emit)/attention_emit(5469行/146 emit/363 fn)/gemm_emit(3788行/69 emit) — 实测已高度分解"
  regressionAssertion:
    - "emit 序列快照测试: 重构前后 VmProgram 指令序列字节级等价"
    - "numerical_sim 等价验证: 重构前后 interpret_vm_program 输出数值一致"
    - "重构后 compile_mega_kernel_vm 入口参数 ≤6, 单 method 行数 ≤500, 圈复杂度 ≤10"
  locations:
    - "src/compiler/codegen/vm/mega_kernel_emit.rs L924-2427 (compile_mega_kernel_vm)"
  rootCause: "JIT emit 入口承载完整编排(prologue+batch_mode+sampling)的过程式 emit 长序列, 25 个编排级状态以游离局部变量穿透三阶段, 无编排层 ctx 收编 → long_method(1503行/CC~88) + long_parameter_list(15参数)"
  归因时间: 2026-07-01
  status: 待激活 ⏸ | residual: N/A(未根治, blocked on P3)
```

---

## BCE-20260630-LOWER-INSTR-GOD-MATCH — lower_instr 巨型 match god-match 根治 (A类, 已闭环)

### smellClass: GOD-MATCH-DISPATCH (巨型 match VmInstr/TraceOp lowering dispatch, 单函数 3000-5000 行)

**宪法依据**: P-2 复杂度限制 (圈复杂度 ≤10) + ARCH-AUTO-INSTR-SELECT (ComputePattern 通用处理器, 禁 per-OpKind 手写) + NO-SILENT-FALLBACK (catch-all 返回 Err 非 NOP)。

**模式签名 (A类)**: 单函数 `match instr { VmInstr::Variant1 => ..., ... 153 arm }` 巨型 dispatch, 每新增 VmInstr arm 膨胀。三 ISA lower_instr (x86 3908行/gpu 5087行/aarch64 3340行) + program/verify/reg_alloc 的 match (renumber/validate/liveness) + auto_select dispatch_trace_op (1213行)。

**根治策略 (A类, architect 裁决 sessionId 2b51725c)**:
- L0 分类 dispatch: `match instr.category() { Memory => lower_memory(), ... 8 类全枚举无 catch-all }`
- L1 变体路由: 8 个 lower_<cat>_<isa> 方法
- L2 叶子 emit: 每变体独立 fn (Python 脚本机械抽取, refactor_code 在 .inc.rs 失效)
- catch-all 返回 Err (NO_SILENT_FALLBACK)
- 共享分类器: `vm_instr_category.rs` VmInstr::category() 跨三 ISA + TraceOp::category()

**根治范围 (5 commit, 已闭环)**:
- P1 (548cbc85): x86 lower_instr 3908→30 行 + vm_instr_category.rs + dispatch.inc.rs (8 L1 + 145 L2)
- P2 (f5e284c4): aarch64 3340→410 + gpu 5090→20 + dispatch.inc.rs (8 L1 + 145/146 L2)
- P3 (2622a221): program.inc.rs 6 函数 + verify.rs 4 函数 + reg_alloc.rs 3 函数 (11 god-match 清零)
- P3b (afa8218b): validate/spill god-match split
- P3c (4ba9958d): auto_select TraceOp category dispatch (过渡, auto_lower_trace 查表化为后续 BCE)

**验证**: golden test 6972 passed 0 failed (diff=0, 行为保持) + arch_insight long_method 9→4 (5 god-match 清零)。

**残留 (后续 BCE, 不搁置)**:
- B类 long_method 4 处: mega_kernel_emit (1493行, BCE-MEGA-KERNEL-EMIT-CTX-REFACTOR blocked) + lower_op (1180行, 已根治范本不动) + numerical_sim exec_op_with_pos (987行) + pipeline emit_fusion_groups (758行) — 过程式长序列, 需 ctx 重构/extract_function
- L2 body 二次抽取: high_cyclomatic L2 叶子 fn body 决策点 (if/return 链), 需进一步细分
- auto_select auto_lower_trace 查表化根治 (ARCH-AUTO-INSTR-SELECT, architect 裁决)

**防复发**: SPEC criterion ARCH-LOWER-DISPATCH-LAYERING (dispatch CC 允许 OCP 扩展点, logic CC ≤10) 待写入 SPEC/02-ARCHITECTURE.md §8。

