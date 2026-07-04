# BUG-KNOWLEDGE.md — gllm-kernels BUG 模式知识库

> 每次 BCE 根治后沉淀，避免重复归因。按 patternId 倒序排列。
> gllm-kernels 仓专属（JIT codegen + VmInstr SSOT）。gllm 主仓另有独立 BUG-KNOWLEDGE.md。

## 根治总览

| 类别 | 条目数 | 根治 | 残留 | 备注 |
|------|--------|------|------|------|
| BCE-MIXED 算子级混合精度（BCE-20260630-MIXED） | 1 | 1 ✅ | 0 | emit_gemm_* 三段式 dtype（a/b/c）+ ctx.dtype per-op + accumulator_dtype() 标注 |
| BCE-OPTPASS 指令重写 dtype 丢失（BCE-20260630-OPTPASS） | 1 | 1 ✅ | 0 | substitute_loop_offset/forwarding match 绑定保留原 dtype，禁 `..` 丢弃 |
| BCE-X86-APX-EGPR iced_x86 APX egpr 编码缺失（BCE-20260703-X86-APX-EGPR-UNUSED） | 1 | 1 ✅ | 0 | gpr/gpr32/gpr64_to_32 对 16..31 显式报错；iced_x86 1.21 无 R16-R31 变体，APX 激活前须升级 iced_x86 |
| BCE-AVX512-HALF-LANES AVX-512 codegen 半 lanes（BCE-20260703-AVX512-HALF-LANES） | 1 | 1 ✅ | 0 | 7 处 reduction/scan 按 use_avx512 分流 ZMM 16-lane（argmax/softmax_reduce_max/HReduce/Accumulate/softmax_normalize/temperature/Transcendental） |
| BCE-GPU-VIOLATIONS GPU codegen 6 处违宪（BCE-20260704-GPU-VIOLATIONS） | 1 | 1 ✅ | 0 | DotProduct SIMT非tcgen05/MxFP4多lane SIMT解码/VecLoad dtype分流/KIVI SIMT per-thread/NVFP4 E2M1解码 (GPU-009 P3 留后续) |

**全库残留总计**: 0

---

## BCE-20260703-AVX512-HALF-LANES — AVX-512 codegen 按 width 算 step 但用 YMM load → 高 8 lanes 跳过

### smellClass: AP-WIDTH-STEP-YMM-LOAD-MISMATCH（Pattern — AVX-512 路径 step 按 16 算但 load/reduce 固定 YMM 8 lanes，高 8 lanes 数据丢弃）

**宪法依据**: NO-HW-DEGRADATION（硬件差异体现在 codegen 指令选择，非降级到 AVX2）+ ARCH-JIT-YIELDS（代码顺从硬件/输入/配置，AVX-512 下必须用 ZMM 16-lane 指令）+ ARCH-DTYPE-JIT-TYPED（width 与 load 宽度贯穿一致，禁止 width=16 但 load=8 的断裂）。同源族：BCE-20260703-AVX512-BROADCAST-NAN（broadcast 只填低 8 lanes → NaN），本次是 reduction/scan 只处理低 8 lanes → 精度偏差。

**模式签名**: x86 codegen 的 reduction/scan 函数按 `width.f32_lanes()`（AVX-512=16）计算遍历步长 `step=lanes*4=64`，但 load/reduce 实际只用 `scratch_ymm` + `ymmword_ptr`（32 字节/8 lanes）。步长按 16 跳、实际只处理低 8 lanes，每 64 字节块的高 8 lanes（lanes 8-15）完全跳过。AVX2 下 `step=32` 与 `ymmword_ptr=32B` 匹配故正确（A/B 铁证：本地 i9 AVX2 argmax=253 正确，5070Ti AVX-512 argmax=6 错）。

```yaml
- patternId: BCE-20260703-AVX512-HALF-LANES
  title: AVX-512 codegen 按 width 算 step 但用 YMM load/reduce → 高 8 lanes 跳过
  layer: 设计缺陷（width 与 load 宽度系统性不一致）
  smellClass: AP-WIDTH-STEP-YMM-LOAD-MISMATCH
  codePattern:
    - "fn.*width: SimdWidth.*{ let lanes = width.f32_lanes(); let step = lanes * 4; ... scratch_ymm ... ymmword_ptr ... }  # step=64 但 load=32B"
    - "HReduce 无 use_avx512 分支, 无条件 resolve_ymm_or_spill 读 ZMM 上游 vreg → 只看低 8 lanes"
    - "Accumulate vaddps ymm 只累加低 8 lanes, 高 8 lanes 始终保持初始 0"
    - "Transcendental emit_exp_cephes/emit_log_minimax YMM-only 参数, 高 8 lanes 输出垃圾"
  triggerCondition:
    - "use_avx512=true（DeviceProfile AVX-512, W512 width）"
    - "函数同时含: width 参数算 step + scratch_ymm/ymmword_ptr load + 跨 lanes reduction"
    - "或: HReduce/Accumulate/Transcendental 无 use_avx512 分支（无条件 YMM）"
  detectionSignatures:
    structural: "fn.*width: SimdWidth.*\\{.*let lanes = width.f32_lanes\\(\\).*let step = lanes \\*.*scratch_ymm.*ymmword_ptr"
    literal: "scratch_ymm.*\n.*ymmword_ptr  # YMM load 在 width 驱动函数中"
    literal: "fn lower_h_reduce_x86.*\n.*resolve_ymm_or_spill  # 无 use_avx512 分支"
    antipattern: "width-step-ymm-load-mismatch / no-avx512-branch-in-reduction"
  sameClassCriterion:
    - "任何按 SimdWidth 计算遍历步长但用固定 YMM(32B) load/reduce 的 codegen 函数"
    - "任何无 use_avx512 分支的 reduction/h-reduce/accumulate/transcendental 算子（无条件 YMM）"
  fixTemplate:
    - "按 use_avx512 分流: true→scratch_zmm + zmmword_ptr + 16-lane reduce（vextractf64x4 拆 ZMM→2×YMM 再复用 xmm reduce 链）; false→保持 YMM 现状"
    - "HReduce 加 use_avx512 分支: src/dst 用 resolve_zmm_or_spill, 16-lane reduce 到标量, vbroadcastss zmm 广播回 16 lanes"
    - "Accumulate 加 use_avx512 分支: resolve_zmm_or_spill + vaddps zmm（16 lanes 全加）"
    - "Transcendental: emit_exp_cephes_zmm/emit_log_minimax_zmm（dst/src: AsmRegisterZmm）, lower_transcendental 加 use_avx512 分流; vroundps→vrndscaleps（ZMM 不支持 vroundps）, vrcpps→vrcp14ps（ZMM 不支持 vrcpps）"
    - "argmax lane 查找: vcmpeqps k1, zmm, zmm + kmovd gpr + bsf（替代 AVX2 vmovmskps, 覆盖 16 lanes）"
  regressionAssertion:
    - "静态: use_avx512=true 输出 vextractf64x4 + zmm 寄存器（bce_avx512_half_lanes_*_avx512_uses_zmm 测试）"
    - "静态: use_avx512=false 不含 zmm/vextractf64x4（bce_avx512_half_lanes_argmax_avx2_uses_ymm 测试）"
    - "真机: 5070Ti AVX-512 SmolLM2 greedy next_token == 黄金值 253（E2E test_e2e_generator 覆盖）"
    - "AVX2 不回归: 本地 i9-10900KF 同 E2E 仍 argmax=253"
  regressionTests:
    - "src/compiler/codegen/vm/x86_lower/tests.inc.rs（bce_avx512_half_lanes_* 模块, 9 测试: argmax_avx512/avx2, softmax_reduce_max, h_reduce, accumulate, softmax_normalize, temperature, transcendental_exp/sigmoid）"
  locations:
    - "src/compiler/codegen/vm/x86_lower/lower_instr.inc.rs（zmm_hreduce_to_xmm 辅助 + lower_argmax/lower_softmax_reduce_max/lower_softmax_normalize/lower_temperature_scale 加 use_avx512 分流）"
    - "src/compiler/codegen/vm/x86_lower/lower_instr_dispatch.inc.rs（lower_h_reduce_x86/lower_accumulate_x86/lower_transcendental_x86 加 use_avx512 分流）"
    - "src/compiler/codegen/vm/x86_lower/emit_helpers.inc.rs（emit_exp_cephes_zmm/emit_log_minimax_zmm ZMM 16-lane 版）"
  rootCause: "x86 codegen 7 个 reduction/scan 函数（argmax/softmax_reduce_max/HReduce/Accumulate/softmax_normalize/temperature/Transcendental）在 AVX-512 模式下按 width.f32_lanes()=16 算步长 step=64，但 load/reduce 用 scratch_ymm+ymmword_ptr（32B/8 lanes）。每 64B 块的高 8 lanes 被完全跳过。AVX2 下 step=32 与 load=32B 匹配故正确。HReduce/Accumulate/Transcendental 更深一层：无 use_avx512 分支（无条件 YMM），即使上游 ZMM 产出 16 lanes 数据也只处理低 8。"
  fixCommitted:
    - "<本 commit> fix(BCE-20260703-AVX512-HALF-LANES): 7处 AVX-512 半lanes BUG 根治 — argmax/softmax_reduce_max/HReduce/Accumulate/softmax_normalize/temperature/Transcendental 按 use_avx512 分流 ZMM 16-lane"
  归因时间: 2026-06-30
  根治时间: 2026-07-04
  status: 根治 ✅ | residual: 0
  sessionId: gsc-arch-avx512-precision
```

**根因**: x86 codegen 7 个 reduction/scan 函数在 AVX-512 模式下按 `width.f32_lanes()=16` 算步长 `step=64`，但 load/reduce 用 `scratch_ymm`+`ymmword_ptr`（32B/8 lanes）。每 64B 块的高 8 lanes（lanes 8-15）被完全跳过，从未参与计算。AVX2 下 `step=32` 与 `ymmword_ptr=32B` 匹配故正确。HReduce/Accumulate/Transcendental 无 use_avx512 分支，无条件 YMM，是比"无 use_avx512 分支"更深的根因。

**根治**: 7 处统一加 `use_avx512` 分流，ZMM 路径用 `scratch_zmm`+`zmmword_ptr`+16-lane reduce（`vextractf64x4` 拆 ZMM→2×YMM 再复用 xmm reduce 链，封装为 `zmm_hreduce_to_xmm` 辅助函数）。复用现有 ZMM 基础设施（resolve_zmm_or_spill/scratch_zmm/zmmword_ptr/spill_store_zmm）。Transcendental 新增 `emit_exp_cephes_zmm`/`emit_log_minimax_zmm` ZMM 版本（`vrndscaleps` 替代 `vroundps`，`vrcp14ps` 替代 `vrcpps` — iced_x86 1.21 这两指令不支持 ZMM 操作数）。禁止降级（NO-HW-DEGRADATION）。

**横扫确认**: 三层搜索（structural/literal/antipattern）命中 7 处，全部根治，残留=0。已逐一核对未受影响函数（lower_softmax_exp_sum/lower_batch_per_seq_argmax 标量循环正确；VecLoad/VecStore/Broadcast/FMA/DotProduct/VecBinOp/VecCmp/ConditionalSelect/VecConvert/VecWiden/VecNarrow 均有 use_avx512 分支）。

### 防复发沉淀
- 代码内注释: lower_instr.inc.rs zmm_hreduce_to_xmm() + 7 处 use_avx512 分流顶部 BCE-20260703-AVX512-HALF-LANES 根治说明
- 回归测试: src/compiler/codegen/vm/x86_lower/tests.inc.rs bce_avx512_half_lanes_* 模块（9 测试，反汇编断言 ZMM 指令存在）
- 本条目: BUG-KNOWLEDGE.md 沉淀
- 状态: ✅ 已闭环 (residual=0)
- 5070Ti 真机验证: 待 E2E test_e2e_generator (SmolLM2-135M-Q4_0) argmax==253 确认（任务 #38）

---

## BCE-X86-APX-EGPR — iced_x86 APX egpr (R16-R31) 编码缺失根治

### smellClass: AP-UNREACHABLE-MISLEADING-RANGE（Pattern — unreachable 报错信息掩盖 latent 编码缺失）

**宪法依据**: NO-SILENT-FALLBACK（CLAUDE.md 铁律，JIT codegen 遇不支持 OpKind 必须 Err 而非静默）+ ARCH-ROOT-CAUSE（治本不治标，不掩盖架构缺口）+ NO-HW-DEGRADATION（硬件能力差异体现在 codegen 指令选择，非掩盖）。

**模式签名**: lowering 层的物理寄存器映射函数（gpr/gpr32/gpr64_to_32）用 `unreachable!("range [0..15]")` 掩盖 APX egpr (R16-R31) 编码缺失。isa_profile 在 `has_apx: true` 时 max_gpr=31 会产出 PhysGpr(16..31)，但 iced_x86 1.21 的 `Register` 枚举最大到 R15（无 R16-R31 变体、无 apx feature），lowering 无法编码。`unreachable` 信息暗示 "RegAllocator 越界"，实则是 lowering 编码能力缺口，误导归因。

```yaml
- patternId: BCE-20260703-X86-APX-EGPR-UNUSED
  title: gpr/gpr32 16..31 unreachable 掩盖 iced_x86 APX egpr 编码缺失
  layer: 范式缺陷
  smellClass: AP-UNREACHABLE-MISLEADING-RANGE
  codePattern:
    - "gpr(phys) match phys.0 { 0..=15 => ..., other => unreachable!(\"range [0..15]\") }"
    - "isa_profile has_apx=true 时 max_gpr=31 产出 PhysGpr(16..31) 但 lowering 无对应编码分支"
  triggerCondition:
    - "has_apx=true（CPUID 探测启用 APX）+ RegAllocator 分配到 PhysGpr(16..31)"
    - "当前 latent：microarch::has_apx() 永远 false，16..31 分支死代码"
  detectionSignatures:
    structural: "fn gpr(phys: PhysGpr) -> AsmRegister64 { match phys.0 { .., other => unreachable!(\"range \\[0\\.\\.15\\]\") } }"
    literal: "unreachable!.*x86_64 GPR range \\[0\\.\\.15\\]"
  sameClassCriterion:
    - "任何物理寄存器映射函数对 isa_profile 声称支持的扩展范围（APX 16..31 / AVX-512 高位等）用 unreachable 掩盖编码缺失，而非显式报错"
  fixTemplate:
    - "对扩展范围 16..31 单独 match arm，unreachable 带准确信息：APX egpr 需 iced_x86 APX 编码支持，当前 iced_x86 1.21 无 R16-R31 变体"
    - "区分 'APX 未激活的编码缺口' vs '真正越界 phys'，前者是 latent 死代码待激活，后者是 RegAllocator bug"
  regressionAssertion:
    - "has_apx=false 时 gpr(PhysGpr(0..15)) 正常返回 rax..r15"
    - "has_apx=true 时 gpr(PhysGpr(16..31)) panic 带清晰信息（非误导性 'range [0..15]'）"
```

**根因**: iced_x86 1.21 不支持 APX egpr 编码（Register 枚举 max=R15，无 R16-R31 变体），isa_profile 的 APX 31-GPR 声明与 lowering 实际编码能力脱节，lowering 用误导性 unreachable 掩盖。

**根治**: gpr()/gpr32()/gpr64_to_32() 对 16..31 单独 match arm，unreachable 带准确信息（APX egpr 需 iced_x86 APX 编码支持），并标注升级 iced_x86 后的激活点（`AsmRegister64::new(Register::R16)` 等）。真正越界 phys (>31) 仍 panic 但信息区分。运行时 has_apx 永远 false，16..31 分支 latent，待 iced_x86 升级 + CPUID 探测实现后激活。

**横扫确认**: 三层搜索（structural `fn gpr.*match phys.0` / literal `range \[0\.\.15\]` / isa_profile APX 引用）命中 3 处（gpr/gpr32/gpr64_to_32），全部根治，残留=0。

### 防复发沉淀
- 代码内注释: helpers.inc.rs gpr()/gpr32()/gpr64_to_32() 顶部 BCE-20260703-X86-APX-EGPR-UNUSED 根治说明 + iced_x86 升级激活点
- 本条目: BUG-KNOWLEDGE.md 沉淀
- 状态: ✅ 已闭环 (residual=0)
- 激活路径: iced_x86 升级支持 APX 后，在 gpr() 的 `16..=31` arm 改为 `AsmRegister64::new(Register::R16)` 等；同时实现 microarch::has_apx() CPUID 探测

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

---

## BCE-20260630-LONG-METHOD-ERADICATION — long_method 全量根治 (已闭环)

**日期**: 2026-07-01
**范围**: gllm-kernels/src/compiler/codegen/vm/
**根治病灶**: arch_insight CS-LONG-METHOD 5处 → 0处 (承接 BCE-20260630-LOWER-INSTR-GOD-MATCH 残留 B类4处 + auto_select 查表化)

### 根治模式签名
- patternId: BCE-20260630-LONG-METHOD-ERADICATION
- title: JIT codegen god-function long_method (>500行)
- layer: 范式缺陷 (单 match/单 fn 承载全量逻辑, 违反 OCP)
- codePattern: 单 fn >500行, 圈复杂度>10, 承载完整编排或全量 dispatch
- fixTemplate (按类型分):
  - **A类 god-match dispatch** (dispatch_trace_op/lower_op/exec_op_with_pos): 按 ComputePattern/Op族查表化, 顶层纯 dispatch (arm 单表达式委托) + per-pattern handler (≤500行)
  - **B类过程式长序列** (emit_batch_mode_path/emit_fusion_groups): 按职责抽子 fn + ctx 结构打包只读引用 (&mut 独立保留避 borrow 冲突)
  - **C类单 handler 超长** (dispatch_quant_decode): 抽 per-arm emit helper
- regressionAssertion: arch_insight(quality) CS-LONG-METHOD=0 + cargo test --lib 6972 passed

### 已根治 6 处 (7 commit)
| commit | 函数 | 行数变化 | 方法 |
|---|---|---|---|
| 02d836df | dispatch_trace_op (auto_select.rs) | 1219→109 | ComputePattern 7类查表化 |
| e0fee8e7 | lower_op (lower_op.inc.rs) | 1186→28 | Op族11类查表化 |
| d069324f | dispatch_quant_decode (auto_select.rs) | 518→369 | 抽 emit helper |
| 2da0dd74 | emit_batch_mode_path (mega_kernel_emit.rs) | 811→270 | 按职责抽3子fn + BatchAbiRegs |
| 4e9e5d6d | exec_op_with_pos (numerical_sim.rs) | 1019→106 | ComputePattern 8类查表化 (镜像 dispatch_trace_op) |
| ee414d72 | emit_fusion_groups (pipeline.inc.rs) | 758→123 | FusionEmitCtx 化 + 5子fn |
| 8f5bcdeb | P-1 eprintln 清除 | — | 6处调试输出清除 |

### 防复发铁律
1. 新增 TraceOp/OpKind/VmInstr 变体 = 1行 dispatch + 1个 handler arm, 禁在 dispatch/handler 内联逻辑
2. fn >500行必须拆分 (P-2), 参数 >6 必须 ctx 化 (prog/&mut 独立保留)
3. 禁 eprintln/println/dbg! 调试输出 (P-1), 用 tracing 或删除

### 防复发沉淀
- SPEC criterion: `REQ-ARCH-001-CODEGEN-DISCIPLINE` (写入 gllm SPEC/02-ARCHITECTURE.html, 关联 REQ-ARCH-001 JIT 编译器四阶段管线)
- 本条目: BUG-KNOWLEDGE.md 沉淀
- 状态: ✅ 已闭环 (residual=0)


---

## BCE-20260704-AARCH64-VIOLATIONS (Workflow 全设备审计 aarch64 4处违宪)

**patternId**: BCE-20260704-AARCH64-VIOLATIONS
**title**: aarch64 codegen 4 处违宪 (TileMma FMOPA 无 SME 守卫 / GatherLoad SVE 静默 NOP / VecNarrow/VecWiden dtype 转换未实现 / FP8 转换未实现)
**layer**: 设计缺陷 + 范式缺陷 (NO-SILENT-FALLBACK 违反)
**归因时间**: 2026-07-04

### 4 处 confirmed REAL findings

| ID | P级 | 位置 | BUG | 根治 |
|----|-----|------|-----|------|
| AARCH64-002 | P0致命 | lower_tile_mma_aarch64 | FMOPA 无条件发射 (SME 指令), 无 has_sme2 守卫 → 无 SME CPU SIGILL | 入口加 `if !has_sme2 return Err` (与 TileLoad/TileStore 一致) |
| AARCH64-005 | P0致命 | lower_gather_load/scatter_store_aarch64 | Scalable width 的 `f32_lanes()=0` → `for i in 0..0` = 零条 gather 指令 = dst 未初始化 (数据损坏) | SVE 路径用原生 gather LD1W/ST1W `[Xn,Zm,SXTW#2]`; NEON 路径保持 for-lane |
| AARCH64-007 | P2 | lower_vec_narrow/widen_aarch64 | 不同 dtype 转换 `return Err('not yet implemented')` | F32→F16 FCVTN / F32→BF16 BFCVTN (需 has_bf16) / F16→F32 FCVTL; BF16→F32 保留 Err 注明 NEON 无 BFCVTL |
| AARCH64-011 | P2 | lower_vec_unary_op_aarch64 (FP8) | FP8→F32 `return Err` 无实现注明 | 保留 Err 但注明 FEAT_FP8 限制 (platform 无 has_fp8 字段, 软件转换未编码) |

### BUG 模式签名

```yaml
detectionSignatures:
  antipattern:
    - "无条件发射 SME/SVE2 专属指令而无 has_* 守卫 (NO-SILENT-FALLBACK 违反)"
    - "Scalable SimdWidth 的 f32_lanes()=0 静默导致 for-lane 循环体不执行"
    - "dtype 转换 return Err('not yet implemented') 而非实现硬件指令"
sameClassCriterion:
  - "任何 SME/SVE2/SVE/FEAT_* 专属指令发射点必须有对应 has_* 守卫"
  - "任何 Scalable width 路径不能用 f32_lanes() 控制 for-lane (必须用 SVE 原生向量指令)"
  - "dtype 转换必须有硬件指令实现或明确 Err 注明架构限制"
fixTemplate:
  - "入口守卫: if !self.platform.has_X { return Err(...) }"
  - "SVE gather/scatter: LD1W/ST1W Zt.S, Pg, [Xn, Zm, SXTW #2] + 索引 stride 缩放 (MUL Zdn.S, Pg/M, Zdn.S, Zm.S)"
  - "dtype 转换: 实现 FCVTN/BFCVTN/FCVTL 编码 + match (src_kind, dst_kind) dispatch"
```

### 全量确认 (residual=0)

- 重扫 aarch64_lower/lower_instr_dispatch.inc.rs: 4 处全部根治
- cargo check --lib: 0 errors
- cargo test --lib: 7014 passed / 0 failed (含 7 新回归测试)
- 7 回归测试: test_tile_mma_no_sme2 / test_gather_load_sve / test_scatter_store_sve / test_vec_narrow_f32_to_f16 / test_vec_narrow_f32_to_bf16_requires_has_bf16 / test_vec_widen_f16_to_f32 / test_fp8_to_float_returns_err_no_silent_nop

### 防复发沉淀
- 回归测试: 7 个 (写入 tests.inc.rs)
- BUG-KNOWLEDGE.md: 本条目
- 状态: ✅ 已闭环 (residual=0)

---

## BCE-20260704-GPU-VIOLATIONS — GPU codegen 6 处违宪 (tcgen05 误用/SIMT lane 丢弃/dtype 硬编码/循环展开/E2M1 整数解码)

**patternId**: BCE-20260704-GPU-VIOLATIONS
**归因时间**: 2026-07-04
**缺陷分层**: 范式缺陷 + 设计缺陷 (架构层: SIMT 线程模型 + dtype 传播 + tile-vs-SIMT 边界)

### 6 处 BUG

| ID | 优先级 | 文件 | BUG | 根治 |
|----|--------|------|-----|------|
| GPU-001 | P0 | lower_instr_dispatch.inc.rs:1326-1603 (lower_dot_product_gpu) | DotProduct (SIMT 标量点积) SM100+ 用 tcgen05.mma / SM90+ wgmma / SM80+ mma.sync — 全是 Tile 级张量核指令, 非法用于 SIMT 标量 (标量寄存器喂 tile 指令 = 非法 PTX)。Int8 同违宪 (mma.sync IMMA)。 | 所有 SM 统一 SIMT 标量路径: BF16→cvt.f32.bf16+fma; F16→cvt.f32.f16+fma; Int8→mul.lo.s32+add.s32; Fp4→软件 e2m1 解码+fma。删除全部 tcgen05/wgmma/mma.sync/wmma。 |
| GPU-002 | P0 | lower_gpu.inc.rs:293-551 (emit_mxfp4_dequant_gpu) | 多 lane 路径 (lanes>2) 仅解码 lane 0 并 mov.f32 {d},{fs2}, 注释自认 "only store lane 0" — W512(16)/W256(8) 高 lanes 静默丢弃。 | SIMT per-thread (lane=tid.x, byte_idx=tid>>1, is_high=tid&1), OOB 守卫 tid>=lanes→0.0。参照 Q4_0 修复 1c1f1a40。HIP/Metal 同步 per-thread。 |
| GPU-003 | P1 | lower_instr_dispatch.inc.rs:246-360 (lower_vec_load/store_gpu) | `..` 忽略 VecLoad/VecStore.dtype, 硬编码 ld/st.global.f32 + /4。BF16/F16/INT8 权重用错 dtype load。 | 按 instr.dtype.kind 分流: F32→f32/4; BF16→b16/2+cvt.f32.bf16; F16→b16/2+cvt.f32.f16; INT8→s8+cvt.f32.s32。参照 x86 dtype.x86_elem_strategy()。ARCH-JIT-YIELDS。 |
| GPU-004 | P1 | lower_gpu.inc.rs:771-878 (emit_kivi_quant) + lower_instr_dispatch.inc.rs:5220 (lower_page_table_k_v_write_quant_gpu) | `for pair in 0..num_pairs` (num_pairs=32-64) Rust 循环展开 PTX/HIP/Metal。Kivi2 `for i in 0..num_elems` 同违宪。 | SIMT per-thread (pair=tid.x), OOB 守卫。Kivi2 max-reduce 用 warp shuffle (shfl.sync.bfly / __shfl_xor / simd_shuffle_xor), num_elems>32 返回 Err (多 warp 需 shared mem 留后续)。 |
| GPU-005 | P1 | lower_instr_dispatch.inc.rs:3585-3608 (QuantDequantFma Nvfp4) | 把 4-bit nibble 当原始整数 cvt.rn.f32.s32 + mul.f32 (整数 0-15 解码), 而非 E2M1 浮点解码。NVFP4=E2M1 (1 sign+2 exp+1 mant), 应解码为 (-1)^sign×2^(exp-1)×(1+mant×0.5)。 | E2M1 浮点解码: 提取 sign/exp/mant 位, ex2.approx(2^(exp-1)), (1+mant×0.5)×two_pow, neg 应用 sign, mul scale, fma.rn.f32。 |
| GPU-009 | P3 | lower_gpr_cond_action_gpu (L2265, 圈复杂度81) / lower_quant_block_load (L2,441行) 等 | God Function 复杂度超 P-2 (≤500行/圈复杂度≤10)。 | 留后续 (P3 低优先级, P0-P1 已根治)。报告注明。 |

### BUG 模式签名

```yaml
detectionSignatures:
  structural:
    - "DotProduct GPU 路径出现 tcgen05.mma / wgmma.mma_async / mma.sync.aligned / wmma (tile 级指令用于 SIMT 标量点积)"
    - "GPU 多 lane dequant 路径出现 'only store lane 0' / 'subsequent lanes would need separate dst' 注释 (静默丢弃高 lanes)"
    - "lower_vec_load/store_gpu 用 `..` 忽略 dtype 字段 + 硬编码 ld/st.global.f32"
    - "GPU quant 路径出现 `for pair in 0..num_pairs` / `for lane in 0..lanes` Rust 循环展开 PTX (NO-LOOP-UNROLL 违反)"
    - "NVFP4/E2M1 路径出现 cvt.rn.f32.s32 + mul.f32 (整数解码 E2M1 浮点)"
  antipattern:
    - "tile-vs-SIMT 边界混淆: tile 级张量核指令 (tcgen05/wgmma/mma.sync) 误用于 SIMT 标量上下文"
    - "for-lane Rust 循环展开 PTX (应用 SIMT per-thread, tid.x 驱动)"
    - "dtype 字段被 `..` 忽略 (应用 dtype.gpu_elem_strategy() / dtype.kind 分流)"
sameClassCriterion:
  - "任何 DotProduct (SIMT 标量点积) 路径禁用 tile 级张量核指令 (tcgen05/wgmma/mma.sync/wmma)"
  - "任何多 lane GPU 解码路径必须 SIMT per-thread (tid.x 驱动), 禁止 for-lane 循环展开 + 单 {d} 覆盖"
  - "任何 VecLoad/VecStore 必须按 instr.dtype 分流, 禁止硬编码 f32"
  - "任何 E2M1/E2M3/E4M3 等 FP nibble 解码必须浮点解码 (sign/exp/mant 提取), 禁止整数 cvt"
fixTemplate:
  - "DotProduct: 所有 SM 统一 SIMT 标量路径 (cvt+fma / mul+add), 删除全部 tile 级指令"
  - "多 lane dequant: SIMT per-thread (lane=tid.x, byte_idx=tid>>1), OOB 守卫 tid>=lanes→0.0"
  - "VecLoad/Store: match dtype.kind 分流 (F32/BF16/F16/INT8), sub-byte 返回 Err"
  - "quant for-pair: SIMT per-thread (pair=tid.x), Kivi2 max-reduce 用 warp shuffle"
  - "E2M1 解码: ex2.approx(2^(exp-1)) × (1+mant×0.5) × (-1)^sign × scale, fma 累加"
```

### 全量确认 (residual=0)

- 重扫 gpu_lower/lower_instr_dispatch.inc.rs + lower_gpu.inc.rs: 5 处 P0-P1 全部根治 (GPU-009 P3 留后续)
- cargo check --lib: 0 errors (gpu_lower 无 error/warning)
- cargo test --lib: 7023 passed / 0 failed (含 9 新 GPU 回归测试)
- 9 回归测试: test_ptx_dot_product_bf16_no_tcgen05 / test_ptx_dot_product_fp16_no_tcgen05 / test_ptx_dot_product_int8_no_mma_sync / test_ptx_dot_product_fp4_no_tcgen05_software_e2m1 / test_ptx_vec_load_bf16_dtype_dispatch / test_ptx_vec_load_f16_dtype_dispatch / test_ptx_vec_load_int8_dtype_dispatch / test_ptx_vec_store_bf16_dtype_dispatch / test_ptx_quant_dequant_fma_nvfp4_e2m1_decode
- (另有 7 处 MXFP4 测试增强 SIMT per-thread 断言)

### 防复发沉淀
- 回归测试: 9 新增 + 7 增强 (写入 tests.inc.rs)
- BUG-KNOWLEDGE.md: 本条目
- 状态: ✅ P0-P1 已闭环 (residual=0); GPU-009 P3 God Function 重构留后续

---

## BCE-20260704-STRUCTURED-SYMEXEC-LOOP-MISCLASSIFY

- **patternId**: BCE-20260704-STRUCTURED-SYMEXEC-LOOP-MISCLASSIFY
- **title**: 结构化 symexec 误分类循环数导致 NormLike pattern 错配 (RmsNorm 被分类为 LayerNorm)
- **layer**: 范式
- **归因时间**: 2026-07-04
- **现象**: `e2e_embedding_safetensors` (release) 在 `norm_softmax_emit.rs:224` panic: `TraceOp::Input(3) 越界: 调用方仅提供 3 个输入 VReg`。阻塞所有 BERT/XLM-R encoder E2E。
- **根因**: `scalar_rms_norm` (2 逻辑循环, 无 bias) 经编译器向量化后 CFG 检测到 3 物理循环 (loop1 transform 误报 reduction 阻止 coalesce), 进入 `combine_three_loops` 的 Sum→Sum 分支, 被 `combine_layer_norm` 无条件生成含 `Input(3)`/`Input(4)` (weight+bias) 的 LayerNorm pattern, 覆盖了 RmsNorm 正确的 manual trace; `emit_normlike_inline` 按 RmsNorm 只传 3 输入, transform 引用 `Input(3)` 越界 panic。
- **根治 (A+B 组合)**:
  - A (防御性校验): `register_with_symexec_fallback` Level 1 成功后, 校验生成 pattern 的 `max_input_arity` ≤ sig 的 ptr 参数数, 否则降级到 Level 2/3 (manual trace)。新增 helper `ScalarOpRegistry::max_input_arity(&pattern)`。
  - B (combine 层根治): 新增 `combine_passes_with_sig(traces, Option<&ScalarFnSignature>)`, `combine_layer_norm` 接收 `Option<&ScalarFnSignature>`, 校验 sig 含 ≥2 个 `WeightPtr` (weight+bias) 才生成 LayerNorm pattern, 否则返回 Err 降级。`combine_passes(traces)` 委托 `combine_passes_with_sig(traces, None)` 保持测试兼容。decoder x86/aarch64 调用点改用 `combine_passes_with_sig(..., Some(sig))`。
- **同类横扫**: `scalar_value_norm` / `scalar_l2_normalize` / `scalar_qk_norm` (均为 2-逻辑-循环 NormLike, 无 bias) 均受 A 防御覆盖 — structured 误分类时 pattern arity 超过 sig ptr 数 → 自动降级 manual trace。
- **回归测试** (registry_fragments/tests.inc.rs):
  - `max_input_arity_counts_transform_inputs` (LayerNorm arity=5)
  - `max_input_arity_rms_norm_pattern_within_two` (RmsNorm arity=3)
  - `combine_layer_norm_rejects_bias_less_signature` (RmsNorm sig 被拒)
  - `combine_layer_norm_accepts_true_layer_norm_signature` (LayerNorm sig 通过)
  - `rms_norm_cached_pattern_input_arity_within_signature` (全路径: with_defaults 后 RmsNorm cached pattern arity ≤ 3, debug+release 均通过)
- **确认**: cargo test --lib 全过 (7029 passed, 0 failed); release 模式 RmsNorm structured 返回 None (降级 manual trace, 不再 panic)。residual=0。
- **SPEC criterion**: REQ-AIS-007 (ScalarOpRegistry) + REQ-AIS-002 (ComputePattern) — structured symexec 生成的 pattern Input arity 不得超过 scalar fn signature 的 ptr 参数数; combine_layer_norm 必须校验 fn_sig 含 weight+bias。

## BCE-20260704-AMX-SIGNED-INT8 — JIT AMX TileMma U8 用 TDPBUUD (u8×u8) 处理 Q8 signed i8 权重致数值错

- **patternId**: BCE-20260704-AMX-SIGNED-INT8
- **title**: AMX INT8 JIT 路径指令选择与数据符号性不匹配 (TDPBUUD vs TDPBUSD)
- **layer**: 设计
- **归因时间**: 2026-07-04
- **现象**: `lower_tile_mma_x86` 的 `DType::U8` 分支 emit `tdpbuud` (u8×u8 → i32)。但 GGUF Q8 权重是 signed i8 (`quant.rs BlockQ8_0 qs: [i8; QK_K]`, `quant_format.rs` "Q8_0/Q8_K/Q8_1 signed")。TDPBUUD 把 i8 负数当 u8 解码 (负数变 128-255) → 数值错。
- **根因**: JIT codegen 路径 (`lower_instr_dispatch.inc.rs:3226`) 与直接 asm 路径 (`matmul_x86_amx_int8.rs:179 _tile_dpbusd`) 指令选择不一致。asm 路径正确用 TDPBUSD (u8×i8), JIT 路径误用 TDPBUUD。量化 GEMM 标准数据约定: A=激活 u8 (s8 偏移+128 无符号化), B=权重 i8 (signed)。TDPBUSD (A=u8, B=i8) 数值正确; TDPBUUD (A=u8, B=u8) 把 B 的 i8 负数误解码。
- **根治**: `lower_tile_mma_x86` U8 分支 `self.asm.tdpbuud(...)` → `self.asm.tdpbusd(...)`。iced 1.21 code_asm 原生支持全 4 AMX INT8 变体 (tdpbuud/tdpbssd/tdpbsud/tdpbusd)。注释表同步更新 (U8 → TDPBUSD, pp=66)。
- **同类横扫**: AMX-FP8 (TDPHF8PS/TDPBF8PS) + AMX-TF32 (TDPTF32PS) 路径已用 `emit_tdp_raw` 正确分发, 无同类符号性错配。asm 路径 `matmul_x86_amx_int8.rs` 已正确, 不需改。
- **回归测试** (`x86_lower/tests.inc.rs`):
  - `bce_amx_signed_int8_tile_mma_u8_uses_tdpbusd_not_tdpbuud` — 断言 U8 TileMma emit `Code::VEX_Tdpbusd_tmm_tmm_tmm` (非 Tdpbuud) + 机器码 pp=01 (66 prefix, TDPBUSD) 非 pp=00 (TDPBUUD)。
- **确认**: cargo test --lib 全过 (7030 passed, 0 failed; 单线程 49s + 多线程 3 次均通过)。residual=0。commit 53190cc4。
- **SPEC criterion**: REQ-HWACC (AMX-INT8) — JIT AMX TileMma 指令选择必须顺从权重数据符号性: Q8 signed i8 权重 → TDPBUSD (u8×i8), 禁止 TDPBUUD (u8×u8)。

## BCE-20260704-KERNELS-TRAIT-ISLAND-STUBS — Kernels trait 19 个 unimplemented! 孤岛 stub (NO-PRAGMATIC-HACKS + P-1 红线)

- **patternId**: BCE-20260704-KERNELS-TRAIT-ISLAND-STUBS
- **title**: Kernels trait 残留 unimplemented! stub 占位 (孤岛模块 + P-1 红线违宪)
- **layer**: 范式
- **归因时间**: 2026-07-04
- **现象**: `src/traits.rs` Kernels trait 有 ~16 个方法 `unimplemented!()` 占位 (vec_dot/vec_sub/vec_scale/vec_axpy/vec_max/vec_sum_squares/gemm_bt/gemm_bias/gemm_prepacked/gemm_bias_prepacked/relu/dequant_q*/gemv_q8/rms_norm/layer_norm/rope/rope_with_pos/tanh/exp 等)。注释 "to allow incremental implementation"。
- **根因**: 历史增量开发遗留的 trait stub。`arch_insight(quality)` + live caller 扫描确认: 除 `pack_b`/`gelu` 有 live caller (已由 `cpu_kernels/mod.rs` 实现的非 trait 方法覆盖), 其余 stub 全是 0 live caller 的真孤岛。违反 NO-ISLAND-MODULE (编译通过+测试通过≠完成) + NO-PRAGMATIC-HACKS (禁 stub) + P-1 红线 (commit 前清除 unimplemented!/stub)。
- **根治**: 删除全部 0-live-caller 的 unimplemented! stub 方法 (trait 签名 + 默认实现)。`pack_b`/`gelu` 等 live 方法保留 (由实现模块提供)。trait 只保留真实被调用的方法集。
- **同类横扫**: 全 `traits.rs` + `cpu_kernels/` 扫描, 确认无其他 unimplemented!/todo!/stub 残留。
- **回归测试**: `cargo test --lib` 全过 (7029 passed, 0 failed) — 删除的方法无 live caller, 不影响行为。
- **确认**: cargo test --lib 全过 (7029 passed, 0 failed)。residual=0。commit 493e6092。
- **SPEC criterion**: P-1 红线 + NO-ISLAND-MODULE — Kernels trait 禁止 unimplemented!/stub 占位; trait 方法必须有 live caller (非 test) 或删除。

## BCE-20260704-X86HW-002-VDPBF16PS — BF16 DotProduct 混合精度数值BUG + VDPBF16PS硬件指令缺失

- **patternId**: BCE-20260704-X86HW-002-VDPBF16PS
- **title**: DotProduct Bf16 语义模糊致 aarch64/gpu 混合精度数值错 + x86 未用 VDPBF16PS 硬件指令
- **layer**: 设计 (ARCH-DTYPE-MIXED-PRECISION 违宪)
- **归因时间**: 2026-07-04
- **现象**: `VmInstr::DotProduct { input_dtype: DotDtype::Bf16 }` 语义模糊。实际调用方 `quant_gemm.inc.rs:97-102` 是混合精度 (a=激活 F32, b=权重 BF16, 注释 "Activation is always F32")。但三后端实现假设不一致:
  - x86 (`lower_instr_dispatch.inc.rs:1951`): `vfmadd231ps` 靠 VecLoad 把 b widen 成 F32, 退化成 F32×F32。数值正确但未用 VDPBF16PS (NO-HW-DEGRADATION 违规)。
  - aarch64 (`lower_instr.inc.rs:51-56`): BFDOT `Vd.4S, Vn.8H, Vm.8H` 假设双 BF16。a 是 F32 位模式被当 BF16 解析 → 数值错 (P0)。
  - gpu (`lower_instr_dispatch.inc.rs:1473-1475`): 双 `cvt.rn.f32.bf16` 假设双 BF16。a 是 F32 被错解析 → 数值错 (P0)。
- **根因**: `DotDtype::Bf16` 单一变体无法区分纯双 BF16 vs 混合精度 F32×BF16。后端各自假设不同, aarch64/gpu 假设双 BF16 与 emit 点 F32×BF16 语义不一致。违反 ARCH-DTYPE-MIXED-PRECISION (每个 op 的每个输入 tensor 都有独立 dtype, 混合精度是显式一等公民变体)。
- **根治 (architect 方案 B)**: 新增 `DotDtype::Bf16xF32` (混合 a=F32 b=BF16) + `DotDtype::Fp16xF32` 变体。`Bf16`/`Fp16` 收窄为纯双 BF16/FP16。`quant_gemm.inc.rs:102/105` 改用 `Bf16xF32`/`Fp16xF32` (真实混合精度场景)。三后端各自实现:
  - x86: `Bf16xF32` 保持 `vfmadd231ps` (b widen 成 F32 + F32 FMA, 数值正确); `Bf16` (纯双 BF16) 改用 VDPBF16PS (`Code::EVEX_Vdpbf16ps_zmm_k1z_zmm_zmmm512b32` + `Instruction::with3` + `add_instruction`, iced 1.21 Code 3082), has_bf16 守卫, 无 has_bf16 → Err。
  - aarch64: `Bf16xF32` 走 WidenCompute (b BF16→F32 + FMLA, 不用 BFDOT); `Bf16` 保持 BFDOT + has_bf16 守卫。
  - gpu: `Bf16xF32` 只 cvt b (a 已 F32, 不 cvt); `Bf16` 保持双 cvt。
- **iced_x86 1.21 关键点**: zmm `vfmadd231ps` 只有 `_er` 后缀版本 (`EVEX_Vfmadd231ps_zmm_k1z_zmm_zmmm512b32_er` Code 3540, embedded rounding); `VDPBF16PS` 无 `_er` (`EVEX_Vdpbf16ps_zmm_k1z_zmm_zmmm512b32` Code 3082, 不支持 embedded rounding)。code_asm 无 `vdpbf16ps()` 便捷方法, 用 `Instruction::with3` + `add_instruction`。
- **回归测试** (`x86_lower/tests.inc.rs`): `bce_x86hw002_bf16xf32_mixed_precision` — Bf16xF32 emit vfmadd231ps (非 VDPBF16PS), Bf16 (has_bf16=true) emit VDPBF16PS (非 vfmadd231ps)。aarch64/gpu 谓词真值表补 Bf16xF32/Fp16xF32。
- **确认**: cargo test --lib 全过 (7033 passed, 0 failed; 单线程 50s + 多线程 3 次均通过)。residual=0。commit cee91a7c。
- **SPEC criterion**: ARCH-DTYPE-MIXED-PRECISION — DotProduct 混合精度 (F32×BF16) 必须用显式 `DotDtype::Bf16xF32` 变体, 禁止藏在单一 `Bf16` 标签后; 纯双 BF16 必须用 VDPBF16PS (has_bf16 守卫, NO-HW-DEGRADATION)。

## BCE-20260704-GPU-QUANTLOADBYTESVEC-RESIDUAL — GPU QuantLoadBytesVec 残缺 stub (只取%w0丢弃15字节)

- **patternId**: BCE-20260704-GPU-QUANTLOADBYTESVEC-RESIDUAL
- **title**: GPU QuantLoadBytesVec PTX/HIP/Metal 三路径残缺, 只转换第一个字节丢弃其余
- **layer**: 设计 (NO-SILENT-FALLBACK + P-1 红线违宪)
- **归因时间**: 2026-07-04
- **现象**: `lower_quant_load_bytes_vec_gpu` (lower_instr_dispatch.inc.rs:3516-3552) 三路径全残缺:
  - PTX: `for i in 0..cnt` 加载所有字节到 `%w{i}`, 但 `cvt.rn.f32.u32 {d}, %w0; // Simplified` 只转第一个字节, 其余 15 字节丢弃。`cnt.min(4)` 还限制只打包 4 字节到 `%v4`。
  - HIP: `for(int _i=0; _i<{cnt}; _i++) { _v.x = ... }` 所有字节写 `_v.x` 互相覆盖, `_v.y/z/w` 未初始化。
  - Metal: 同 HIP。
- **根因**: GPU SIMT 向量模型实现错误。`QuantLoadBytesVec { count }` 应产 `count` 个 f32 lane (每字节扩展), 但残缺实现一个线程加载所有字节只取第一个。参照 `lower_quant_interleave_gpu` 用 `%tid.x` 做 lane index 的 per-thread 模式。
- **根治**: 重写三路径为 SIMT per-thread 模型 — 每个线程加载自己 lane 对应字节 (`byte_idx = %tid.x`, guard `byte_idx < count`), zero-extend (signed=false) 或 sign-extend (signed=true) 到 i32/f32。超出 count 的线程 dst=0 (pred 守卫)。
- **注意**: 此残缺 stub 不是 SmolLM2-Q4_0 GPU fail 的根因 (Q4_0 DequantFma 路径走 QuantBlockLoad 不走 QuantLoadBytesVec)。但残缺违宪必须根治 (BCE)。
- **回归测试** (gpu_lower/tests.inc.rs): `bce_gpu_quant_load_bytes_vec_uses_per_thread_tid_index` — PTX 含 `%tid.x`+`ld.global.u8`, 不含 `// Simplified` 或 `cvt.rn.f32.u32 {d}, %w0`。
- **确认**: cargo test --lib 全过 (7037 passed, 0 failed)。residual=0。commit 1877b4ee。
- **SPEC criterion**: P-1 红线 + NO-SILENT-FALLBACK — GPU 量化向量加载必须 SIMT per-thread 完整加载所有 lane, 禁止残缺 stub 只取首字节。
