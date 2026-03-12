# gllm-kernels

**极限性能 CPU 算子库** — 逼近硬件理论峰值的底层计算原语。

> **定位**：纯算子库（Operator Library），不含任何业务逻辑（无 Attention、无 KV Cache、无推理流程）。上层推理引擎通过组合这些算子构建完整推理管线。

---

## 优先级铁律（PRIORITY HIERARCHY）

| 优先级 | 原则 | 含义 |
|--------|------|------|
| **P0 🔴 逼近理论极限** | 每个算子尽可能逼近硬件理论峰值 | compute-bound 算子逼近 FLOPS 峰值；memory-bound 算子逼近带宽峰值 |
| **P1 🟡 JIT 编译器自动优化** | 所有算子通过编译器融合决策 + 程序化代码生成达到最优 | Phase 3 用 iced-x86/dynasm-rs 生成每条指令，自动 epilogue injection / loop fusion / tile-level fusion |
| **P2 🟢 代码量最少** | 编译器代码本身保持精简 | 宏/泛型复用编译器内部逻辑，避免重复代码 |
| **P3 ⚪ 可维护性** | 新增 ISA/量化格式/算子的变更路径清晰 | 遵循维护检查清单 |

> **核心判断准则**：所有性能优化通过 JIT 编译器实现。算子的唯一定义来源是 `extern "C"` 纯标量函数，编译器通过二进制符号执行自动提取计算结构（OpTrace），然后根据 DeviceProfile 生成最优融合 SIMD 代码。项目中现有的手写 asm / intrinsics / 宏生成实现作为正确性基准和性能参考。

---

## 性能目标（PERF-TARGET）🚨 铁律

### 理论峰值计算方法

| 瓶颈类型 | 理论峰值公式 |
|----------|-------------|
| **Compute-bound** (GEMM) | `核心数 × 频率 × FMA吞吐 × SIMD宽度 × 2` |
| **Memory-bound** (GEMV, 激活, 归一化) | `内存带宽 / (输入+输出字节数)` |
| **量化 GEMV** | `min(计算峰值, 带宽/量化字节数)` |

### 参考对标

| 库 | 典型效率 | 我们的目标 |
|---|---|---|
| Intel MKL (GEMM) | 85-95% | 逼近 |
| OpenBLAS (GEMM) | 70-85% | 超越 |
| llama.cpp (量化 GEMV) | 60-75% | 超越 |

### 当前状态

| 算子 | 当前效率 | 达成路径 |
|------|---------|---------|
| F32 GEMM | unpacked ~42%, prepacked ~59% (ASM 微内核路径) | JIT 编译器 Phase 3 自动生成最优代码 |
| 量化 GEMV | intrinsics 路径 | JIT 编译器 Phase 3 自动生成最优代码 |
| Softmax/RMSNorm/SiLU | ALU-limited 7-13 GiB/s | JIT 编译器 Loop Fusion 消除中间 writeback |

> **性能调优路线**：全部算子通过 JIT 编译器 Phase 3 自动生成最优代码，不存在手动调优路径。

---

## SPEC 导航（Single Source of Truth）

| 文件 | 内容 |
|------|------|
| `SPEC/01-REQUIREMENTS.md` | 算子清单 + 性能需求 + 编译器需求 |
| `SPEC/02-ARCHITECTURE.md` | 核心架构：手写汇编 + 宏驱动 + 运行时分发 + §8 编译器架构 |
| `SPEC/03-DATA-STRUCTURE.md` | 数据结构 + 宏架构详细设计 |
| `SPEC/05-LAYER2-INFERENCE.md` | 推理后端 + §7 JIT 编译器流水线 |
| `SPEC/04-GPU-BACKEND.md` | GPU 后端规划（CUDA/Metal）🔴 未实现 |
| `SPEC/PLAN-phase4-maturity.md` | Phase 4 成熟度路线图 |
| `SPEC/PLAN-quality-hardening.md` | 质量加固计划 |

---

## 🚨 算法意图编译器（ARCH-COMPILER）— 最易偏离的设计

> **核心原则：标量定义 → 二进制分析 → 融合决策 → 全新代码生成。**
> **融合 = 全新代码生成。不是 trampoline 调度，不是模板拼接。**

### 四阶段编译流水线

```
ScalarOpRegistry (extern "C" 标量函数)
    │
    ▼
Phase 0: 二进制符号执行
    · iced-x86 Decoder 反汇编标量函数
    · 符号执行提取计算结构 → OpTrace
    · OpTrace = { pattern: ComputePattern, body: Vec<TraceOp> }
    · 首次分析后缓存，同一算子不重复分析
    │
    ▼
CompilerGraph (from GLLM) + DeviceProfile
    │
    ▼
Phase 1: 语义 DAG 构筑
    · 算子 → 查 ScalarOpRegistry → 取已缓存的 OpTrace
    · OpTrace.pattern 自动推导算子分类（不再手动映射）
    · 张量 def-use 链 + 后支配树
    │
    ▼
Phase 2: Profile-Driven 融合决策
    · 后支配树 + TVM 规则 → 融合组划分
    · Profile 约束检查（L1 容量、寄存器压力、消费者数）
    · 三种融合模式:
      - Epilogue Injection: 取消费者 OpTrace.body，在 GEMM 累加器上原地生成 SIMD 指令
      - Loop Fusion: 遍历每个算子的 OpTrace.body 生成单循环
      - Tile-Level Fusion: 前驱 tile 计算嵌入 GEMM MC 循环，结果留在 L1
    │
    ▼
Phase 3: 全新代码生成（iced-x86 / dynasm-rs）
    · 从 OpTrace 的 Vec<TraceOp> 直接映射到 SIMD 指令
    · TraceOp::Add → vaddps, TraceOp::Exp → 多项式逼近指令序列
    · GEMM: 完整 K-loop + FMA 序列 + epilogue 在累加器上原地执行 + store
    · Elementwise: 单循环体，数据在 ymm 寄存器中流过整个算子链
    · 输出: CompiledLayer (mmap RWX)
```

### 🚫 禁止 JIT Codegen 静默降级 (NO_SILENT_FALLBACK)

**铁律：JIT codegen 遇到无法生成代码的 OpKind 必须返回 `Err`，禁止静默 NOP**：

- ❌ `emit_nop_raw()` / `emit_nop_placeholder()` 作为未实现 op 的 catch-all
- ❌ `match _ => Ok(())` 吞掉未知 OpKind
- ❌ `eprintln!("[WARN]...")` + scalar 计算替代 JIT 编译失败
- ✅ 未实现的 op 必须 `Err(format!("codegen not implemented for {:?}", op_kind))`
- ✅ 仅 `Reshape` / `Transpose`（纯元数据 op，不需要计算）允许 NOP

**理由**：NOP placeholder 让编译成功、测试通过，但输出是全零或内存垃圾。这是最危险的 bug — 静默产生错误结果，无法通过常规测试发现。审查发现 aarch64 codegen 中有 8 处 `emit_nop_raw()` catch-all，导致所有非 GEMM op 在 aarch64 上被静默跳过。

### 🚫 绝对禁止的实现模式

| 禁止模式 | 为什么错 | 正确做法 |
|----------|---------|---------|
| `mov rax, trampoline_addr; call rax` | 数据落地内存，融合收益为零 | 从 OpTrace.body 的 TraceOp 序列生成 SIMD 指令 |
| 预编译微内核变体（gemm_silu, gemm_gelu） | 组合爆炸，不可扩展 | Phase 3 从消费者 OpTrace.body 动态生成 epilogue |
| EmitAction::CallGemm / CallElementwise | "调度器"不是"编译器" | MachineCodeEmitter trait 生成新代码 |
| 模板字节拼接（复制 body bytes） | 融合后算法结构变了，不能拼 | 从 OpTrace 的 TraceOp 逐条映射到 SIMD 指令 |
| 手动维护 OpSemanticsKind 映射表 | 新增算子需改编译器内部 | extern "C" 标量函数 + 符号执行自动提取 |

### 正确的 Phase 3 代码结构

```
GEMM + SiLU epilogue（JIT 生成，非模板）:
  prologue
  NC loop:
    pack_b
    MC loop:
      pack_a
      NR loop:
        微内核:
          vxorps ymm0..ymm11          // 累加器清零
          K-loop:                       // 程序化生成 FMA 序列
            vbroadcastss ymm12, [A]
            vmovups ymm14, [B]
            vfmadd231ps ymm0, ymm12, ymm14
            ...
          // ★ epilogue: 从消费者 OpTrace.body 提取 TraceOp 序列
          //   [Neg, Exp, Add(1.0), Recip, Mul] → 逐条映射到 SIMD 指令
          SiLU on ymm0..ymm11 (用 ymm12-14 做 scratch)
          vmovups [C], ymm0..ymm11     // 一次 store
  epilogue
```

### 关键 trait 架构（SPEC §8.6）

```rust
trait PlatformBackend {
    type Emitter: MachineCodeEmitter;
    fn new_emitter(&self) -> Self::Emitter;
}

trait MachineCodeEmitter {
    fn emit_gemm_unit(&mut self, unit: &GemmUnit) -> Result<Vec<u8>>;
    fn emit_fused_loop(&mut self, unit: &FusedLoop) -> Result<Vec<u8>>;
    /// 从 OpTrace.body 的 TraceOp 序列生成 SIMD 指令（对指定寄存器原地执行）
    fn emit_trace_ops(&mut self, ops: &[TraceOp], reg: Register) -> Result<()>;
    fn finalize(self) -> Result<Vec<u8>>;
}
```

### 当前状态

Phase 0（symexec/engine.rs 1104 行）：符号执行引擎已完整实现，支持 FMA/比较/位运算/栈溢出/常量池/libm 识别。生产路径使用手动注入的 OpTrace 作为回退。
Phase 1（graph.rs, semantics.rs, semantic_dag.rs）：CompilerGraph DAG + SemanticDAG + OpClass 自动推导已完成。
Phase 2（fusion.rs + hw_constraints.rs）：5 种融合模式 + HwConstraintChecker + Cost 代价模型 + L1 阈值决策 + elementwise L1 split 全部完成。无缺口。
Phase 3 x86_64（codegen/x86_64.rs 3945 行）：AVX2 完整实现，含 emit_trace_ops_avx2()、emit_trace_on_accumulator()、emit_elementwise_trace_body() 三条路径。
Phase 3 aarch64（codegen/aarch64.rs 546 行）：框架 + TraceOp→NEON 映射已实现，循环结构（GEMM/elementwise）待补全。
详见 `SPEC/02-ARCHITECTURE.md` §8 和 `SPEC/01-REQUIREMENTS.md` §6。

---

## Technology Stack

| Component | Technology | Constraint |
|-----------|------------|------------|
| **Language** | Rust nightly (1.93.0+) | `global_asm!`, `naked_fn`, `target_feature` |
| **JIT 编译器 (主路径)** | iced-x86 (x86_64) / dynasm-rs (aarch64) | iced-x86: Phase 0 反汇编 + Phase 3 代码生成；dynasm-rs: Phase 3 代码生成 |
| **算子定义** | `extern "C"` 纯标量函数 + ScalarOpRegistry | 编译器通过二进制符号执行自动提取 OpTrace |
| **现有算子实现** | `global_asm!` 微内核 + intrinsics + 宏生成 | 正确性基准 + 性能参考（非编译器知识来源） |
| **分发** | `cargo install` 一键安装 | 零外部依赖，纯 Rust crate |

---

## 现有算子实现（ARCH-ASM-MICROKERNEL）

> **定位**：项目中所有现有算子实现（手写 asm、intrinsics、宏生成）作为正确性基准和性能参考。
> 编译器的算子知识来源是 `extern "C"` 纯标量函数 + 二进制符号执行自动提取的 OpTrace。
> 现有 SIMD 实现用于：(1) 正确性回归测试的 golden reference；(2) 性能对标基准。

### 微内核规格

| ISA | 微内核尺寸 | 累加器 | 临时寄存器 | 实现方式 |
|-----|-----------|--------|-----------|---------|
| **AVX2** | 6×16 (6M × 2×ymm) | 12 ymm | 4 ymm | `global_asm!` |
| **AVX-512** | 14×32 (14M × 2×zmm) | 28 zmm | 4 zmm | `global_asm!` |
| **NEON** | 8×12 (8M × 3×v) | 24 v | 8 v | `global_asm!` |

### 运行时 CPUID 分发

```rust
// 启动时一次检测，之后零开销
static ISA: OnceLock<IsaLevel> = OnceLock::new();

fn gemm(a, b, c, m, n, k) {
    match *ISA.get().unwrap() {
        IsaLevel::Avx512 => gemm_avx512_asm(a, b, c, m, n, k),
        IsaLevel::Avx2   => gemm_avx2_asm(a, b, c, m, n, k),
        IsaLevel::Neon   => gemm_neon_asm(a, b, c, m, n, k),
        IsaLevel::Scalar => gemm_scalar(a, b, c, m, n, k),
    }
}
```

---

## 🚨 算子边界定义（ARCH-SCOPE）

### 属于本库的算子（纯计算原语）

| 类别 | 算子 | 瓶颈类型 |
|------|------|---------|
| **BLAS-1** | vec_dot, vec_add, vec_mul, vec_scale, vec_axpy, vec_sum, vec_max | Memory-bound |
| **BLAS-2** | gemv, streaming GEMV (M=1 路径) | Memory-bound |
| **BLAS-3** | gemm, gemm_bias, gemm_prepacked, pack_b, gemm_bt (B-transposed skinny GEMM) | Compute-bound |
| **激活函数** | silu, gelu, relu, tanh, swiglu, softmax, exp | Memory-bound |
| **归一化** | rms_norm, layer_norm | Memory-bound |
| **位置编码** | rope | Memory-bound |
| **量化解码** | dequant_* (18 种格式) | Memory-bound |
| **量化 GEMV/GEMM** | gemv_q4, gemv_q8, gemm_q4, gemm_q8, kquant_matmul, iq_matmul 等 | 带宽/计算混合 |

### 不属于本库的（上层业务）

- ❌ FlashAttention / Paged Attention
- ❌ KV Cache 管理
- ❌ 融合算子（fused_qkv_rope, fused_ffn 等）
- ❌ Embedding lookup
- ❌ Sampling (argmax, top-k, top-p)
- ❌ CUDA/GPU 后端
- ❌ 推理调度、批处理

---

## 🚨 四层宏架构（ARCH-MACRO-LAYERS）

> Layer 1 算子库的内部代码组织。宏批量生成基线实现，手写 asm 提供算子计算结构的参考知识。

```
Layer 1: simd_primitive!     — 硬件原语映射表（每 ISA × 精度 22 个操作）
            ↓ 被调用
Layer 2: define_xxx!         — 算子逻辑模板（基线实现）
            ↓ 被调用
Layer 3: quant_primitive!    — 量化特化原语（位解包/码本查表）
            ↓ 被调用
Layer 4: expand_all_xxx!     — 批量展开

正确性参考 + 算子结构知识来源：
  gemm_avx2_asm()     — 手写汇编 GEMM（编译器据此理解累加器布局、K-loop、store 位置）
  gemv_q4_avx2_asm()  — 手写汇编量化 GEMV（编译器据此理解解包/查表/累加结构）
```

### 路径选择（Layer 1 算子库内部）

- M=1 走 streaming GEMV 路径
- M≤32 走 skinny GEMM intrinsics 路径
- M>32 走 ASM 微内核路径

> **注意**：以上路径选择仅描述 Layer 1 算子库的内部逻辑。生产路径全部由 JIT 编译器 Phase 3 自动生成最优代码。

---

## 🚨 三层零成本分发架构（ARCH-DISPATCH）

```
Layer 1: Backend    → CpuBackend（本库唯一后端）
Layer 2: ISA        → 启动时一次检测（Scalar/AVX2/AVX-512/NEON）— OnceLock
Layer 3: Precision  → 编译时泛型单态化（<E: Element>）— 零开销
```

---

## 🚨 ISA 差异性原则（ARCH-ISA-PERF）

> 不同 ISA 的最优算法**结构不同**，不仅仅是"换指令"。

| 差异维度 | AVX2 (16×256b) | AVX-512 (32×512b) | NEON (32×128b) |
|----------|----------------|-------------------|----------------|
| **GEMM 微内核** | 6×16 手写 asm | 14×32 手写 asm | 8×12 手写 asm |
| **水平求和** | 手动 shuffle 4 步 | 原生 `reduce_add` | 原生 `vaddvq` |
| **f16 计算** | F16C 转换→f32 FMA | AVX512-FP16 原生 FMA | NEON FP16 原生 FMA |
| **INT8 点积** | 无原生支持 | VNNI `vpdpbusd` | `sdot` |

---

## Directory Structure

```
src/
├── lib.rs                  # Crate 入口
├── traits.rs               # Element/Backend/Kernels trait
├── quant.rs                # QuantType 枚举 + 块常量
├── codebooks.rs            # IQ 量化码本常量
│
├── macros/                 # 宏架构
│   ├── simd_primitive.rs   # Layer 1: ISA 原语映射表
│   ├── operator_templates.rs # Layer 2: 算子逻辑模板（基线）
│   ├── quant_primitive/    # Layer 3: 量化特化原语
│   └── expand.rs           # Layer 4: 批量展开
│
├── cpu_kernels/            # CPU 后端
│   ├── mod.rs              # ISA 检测 + 分发
│   ├── scalar/             # Scalar 兜底
│   ├── avx2/               # AVX2（含手写 asm 微内核）
│   ├── avx512/             # AVX-512（含手写 asm 微内核）
│   └── neon/               # NEON（含手写 asm 微内核）
│
├── asm/                    # 手写汇编微内核
│   ├── x86_64/
│   │   ├── gemm_avx2.rs    # AVX2 GEMM 6×16 微内核 (global_asm!)
│   │   ├── gemm_avx512.rs  # AVX-512 GEMM 14×32 微内核 (global_asm!)
│   │   ├── gemm_driver.rs  # 缓存分块驱动 (pack_b + MC/KC/NC blocking)
│   │   └── quant_gemv.rs   # 量化 GEMV
│   └── aarch64/
│       └── gemm_neon.rs    # NEON GEMM 8×12 微内核 (global_asm!)
│
├── compiler/               # 算法意图编译器（JIT）
│   ├── mod.rs              # InferenceCompiler 入口
│   ├── graph.rs            # Phase 1: CompilerGraph DAG ✅
│   ├── semantics.rs        # Phase 1: 算子分类 ✅
│   ├── semantic_dag.rs     # Phase 1: SemanticDAG + OpClass 自动推导 ✅
│   ├── fusion.rs           # Phase 2: 融合决策（5 种模式）✅ 缺口：TileLevelFusion/ComputeRoot 决策
│   ├── hw_constraints.rs   # Phase 2: 硬件约束检查（寄存器/L1/epilogue 深度）✅
│   ├── planner.rs          # Phase 2: ExecutionPlan ✅
│   ├── buffer_alloc.rs     # Phase 2: 张量活性分析 + 区间图着色 ✅
│   ├── parallel.rs         # Phase 2: 并行策略 ✅
│   ├── executable.rs       # CompiledLayer mmap RWX ✅
│   ├── cache.rs            # 编译缓存（内存 + 磁盘）✅
│   ├── ir.rs               # LayerIR 中间表示 ✅
│   ├── trace.rs            # OpTrace / ComputePattern / TraceOp 数据结构 ✅
│   ├── registry.rs         # ScalarOpRegistry（17 个算子注册 + OpTrace 缓存）✅
│   ├── symexec/            # Phase 0: 二进制符号执行引擎
│   │   ├── mod.rs          # 模块入口
│   │   ├── engine.rs       # SymbolicExecutor（1104 行）✅
│   │   └── sym_value.rs    # 符号值类型 ✅
│   └── codegen/            # Phase 3: 代码生成
│       ├── mod.rs          # CodegenOutput
│       ├── emitter.rs      # MachineCodeEmitter trait + CodeGenPlan ✅
│       ├── x86_64.rs       # iced-x86 后端（AVX2 完整，AVX-512 基础）✅
│       └── aarch64.rs      # dynasm-rs 后端（框架 + TraceOp→NEON 映射）🟡
│
├── scalar_ops/             # extern "C" 纯标量函数（算子的唯一定义来源）
│   ├── mod.rs              # register_all() → ScalarOpRegistry ✅
│   ├── activations.rs      # scalar_silu, scalar_gelu, scalar_relu, ... ✅
│   ├── blas.rs             # scalar_gemm, scalar_vec_add, scalar_vec_mul, ... ✅
│   ├── norms.rs            # scalar_rms_norm, scalar_layer_norm, ... ✅
│   └── rope.rs             # scalar_rope ✅
│
└── dispatch/               # 运行时分发
    ├── mod.rs              # device_profile() 全局入口
    └── device_profile.rs   # DeviceProfile + GemmBlocking + IsaLevel ✅
```

---

## Common Commands

```bash
cargo test --lib                      # 运行测试
cargo bench --bench gemm_benchmark    # GEMM 基准测试
cargo bench --bench kernels_benchmark # 全算子基准测试
RUSTFLAGS="-C target-cpu=native" cargo bench  # 启用本机 ISA
```

## Cargo Profile

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

---

## 🚨 JIT Codegen 寄存器生命周期约定（ARCH-JIT-REGS）

> **教训来源**：r13/r14 跨 group 被 clobber 导致 SIGSEGV（pack_a 读取 rdi=0x7e）。

### 跨 Group 寄存器（callee-saved，生命周期 = 整个 CompiledLayer）

| 寄存器 | 用途 | 设置位置 |
|--------|------|---------|
| `r13` | activation input base ptr | `emit_body()` prologue |
| `r14` | weights blob base ptr | `emit_body()` prologue |
| `rbp` | frame pointer | `emit_prologue()` |
| `rbx` | scratchpad base (各 group 内部重新加载) | 各 emit 函数 |

### 铁律

1. **任何 emit_group 实现（BLIS、NormIntoGemm、TileLevelFusion 等）都可能 clobber r13/r14**
2. **主循环通过 push/pop r13/r14 保护跨 group 值**（在 `emit_group_pointer_setup` 之后、`emit_group` 前后）
3. **新增 emit 函数如果使用 r13/r14 作为临时寄存器，无需额外保存**——主循环已保护
4. **[rsp+N] 局部变量不得与 push/pop 冲突**——BLIS 的 `sub rsp, 32` 在 push 之后分配，互不干扰

---

## 🔧 JIT 代码调试方法（DEBUG-JIT）

本地已安装 `lldb-20`（LLDB 20.1.2）和 `rust-lldb`。JIT 生成的机器码无符号信息，需要用以下方法调试：

### 崩溃定位流程

```bash
# 1. 用 LLDB 运行崩溃测试，获取 fault address 和寄存器状态
lldb-20 -- ./target/debug/deps/test_xxx -- --nocapture
(lldb) run
# 崩溃后自动停在 signal handler
(lldb) register read     # 查看所有寄存器
(lldb) disassemble -s $pc -c 40   # 反汇编崩溃点附近代码

# 2. 反汇编 JIT 代码区域（根据 RIP 地址范围）
(lldb) disassemble -s 0x7ffff7e56c00 -c 100

# 3. 关键：对比寄存器值与预期
#    - 指针寄存器应该是大数（0x7fff...），小数值（如 0x7e）= 被 clobber
#    - 循环计数器通常是小整数
```

### 常见 JIT Bug 模式

| 症状 | 可能原因 |
|------|---------|
| 寄存器值是小整数而非指针 | 被循环计数器覆盖（寄存器生命周期冲突） |
| `[rsp+N]` 读到错误值 | 栈局部变量偏移冲突（多个 loc_xxx 用了同一偏移） |
| SIGSEGV 在 pack_a/pack_b | A/B/C 指针未正确设置（检查 rdi/rsi/r8） |
| 第 N 个 group 崩溃，前面正常 | 跨 group 寄存器被前一个 group clobber |
