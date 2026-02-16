# GEMM 优化研究报告

> 本文档是 gllm-kernels CPU GEMM 内核优化的技术研究参考。
> 基于 BLIS、GotoBLAS、oneDNN 等工业级实现的源码分析。

---

## 1. GotoBLAS/BLIS 五层循环与分块参数

### 1.1 五层循环结构（Goto 论文核心）

Goto 的关键洞察：GEMM 的性能瓶颈不是计算，而是数据搬运。五层循环的设计目标是让每层循环的工作集恰好装入对应的 Cache 层级。

```
Loop 5 (jc): N 维度按 NC 分块  → B 的 NC 列 panel 装入 L3
  Loop 4 (pc): K 维度按 KC 分块  → A 的 MC×KC 块装入 L2, B 的 KC×NC 块装入 L3
    Pack B: KC×NC → 连续内存 (B̃)
    Loop 3 (ic): M 维度按 MC 分块  → A 的 MC×KC 块装入 L2
      Pack A: MC×KC → 连续内存 (Ã)
      Loop 2 (jr): N 维度按 NR 分块  → B̃ 的 KC×NR 列在 L1
        Loop 1 (ir): M 维度按 MR 分块  → 微内核: MR×NR 累加器在寄存器
          Micro-kernel: C[MR×NR] += Ã[MR×KC] × B̃[KC×NR]
```

### 1.2 各 ISA 的 BLIS 生产参数（源码实测值）

| 参数 | Haswell (AVX2 f32) | SKX (AVX-512 f32) | Cortex-A57 (NEON f32) | Firestorm (M1 f32) |
|------|--------------------|--------------------|----------------------|---------------------|
| MR | 6 | 32 | 8 | 12 |
| NR | 16 | 12 | 12 | 8 |
| MC | 168 | 480 | 120 | 480 |
| KC | 256 | 384 | 640 | 4096 |
| NC | 4080 | 3072 | 3072 | 9600 |

**关键观察**：
- KC=256 在 Haswell/SKX 上是通用值，与 L1 容量 (32KB) 匹配：`MR × KC × 4B = 6 × 256 × 4 = 6KB` 装入 L1
- M1 的 KC=4096 极大，因为 M1 的 L1D=128KB（是 x86 的 4 倍）
- SKX 的 MR=32 是因为 AVX-512 有 32 个 ZMM 寄存器，可以用更大的 M 维度
- NR 通常是 SIMD 宽度的倍数：Haswell NR=16 = 2×8(AVX2 f32 lanes)

### 1.3 分块参数选择公式

```
KC 约束: MR × KC × sizeof(elem) ≈ L1_size / 2
         （A 的一个 MR×KC 微面板 + B 的一个 KC×NR 微面板装入 L1）

MC 约束: MC × KC × sizeof(elem) ≈ L2_size / 2
         （整个 Ã 装入 L2，留一半给 B̃ 的流式读取）

NC 约束: KC × NC × sizeof(elem) ≈ L3_size / (num_cores)
         （B̃ 在 L3 中被所有核共享）
```

### 1.4 与当前 gllm-kernels 实现的对比

当前实现 (`matmul_x86.rs`):
- AVX-512: TM=14, TN=32 (NV=2, LANES=16), KC=256 ✅
- AVX2: TM=6, TN=16 (NV=2, LANES=8), KC=256 ✅
- 只有 B 矩阵 packing ✅
- 没有 A 矩阵 packing（见 §2）
- 没有 MC 分块（M 维度直接遍历）⚠️
- 没有 NC 分块（N 维度直接遍历）⚠️
- 没有多线程 ⚠️

**差距分析**：当前实现相当于 Goto 五层循环中的 Loop 4 + Loop 2 + Loop 1（有 KC 分块和微内核），但缺少 Loop 5 (NC) 和 Loop 3 (MC) 的显式 Cache 分块。对于 LLM 推理中的典型矩阵尺寸（M 小、K/N 大），KC 分块是最关键的，当前实现已覆盖。

---

## 2. A 矩阵 Packing 的权衡

### 2.1 为什么 BLIS 要 Pack A

在经典 GEMM 中，A 矩阵按行存储，微内核需要从 A 的不同行加载元素（broadcast）。如果 A 的 leading dimension 很大，相邻行的元素在内存中相距很远，导致：
1. TLB miss：跨页访问
2. Cache line 浪费：每次加载 64B cache line 只用 4B（一个 f32）
3. 预取困难：访问模式不连续

Pack A 将 MC×KC 的子矩阵重排为连续的 MR×KC 微面板，使微内核的访问完全顺序。

### 2.2 什么时候可以跳过 Pack A

**LLM 推理的特殊性**：
- Decode 阶段：M=1（单 token），A 就是一个行向量，天然连续，Pack A 无意义
- Prefill 阶段：M=seq_len（可能几百到几千），Pack A 有价值
- 小 batch：M < MR 时，Pack A 的开销大于收益

**经验法则**：
- M ≤ MR：跳过 Pack A（当前实现的 broadcast 方式已经最优）
- M > 4×MR 且 K > 256：Pack A 有 10-20% 收益
- M > 16×MR：Pack A 收益显著（20-40%），因为 TLB 压力成为瓶颈

### 2.3 当前实现的 A 访问模式分析

```rust
// 当前代码中 A 的访问：
let va = simd_primitive!($isa, $elem, splat, *ac.add(k_size * $R));
```

这是 broadcast 模式：从 A 的第 R 行、第 k 列加载一个标量并广播。步长是 `k_size`（A 的列数）。

- 当 k_size 较小（<1024）：步长小，cache line 复用好，不需要 Pack A
- 当 k_size 很大（>4096）：步长大，TLB 压力增加，Pack A 有价值

### 2.4 建议

对于 gllm-kernels 的 LLM 推理场景：
1. **Decode (M=1)**：不需要 Pack A，当前实现已最优
2. **Prefill (M>32)**：添加可选的 Pack A 路径，条件触发
3. **权重矩阵 B**：始终 pre-pack（当前已实现 `pack_b` 和 `matmul_prepacked`）✅

---

## 3. AVX-512 VNNI、BF16 和 AMX 指令

### 3.1 AVX-512 VNNI (Vector Neural Network Instructions)

**指令**：`vpdpbusd` / `vpdpwssd` / `vpdpbusds` / `vpdpwssds`

**核心操作**：
```
vpdpbusd zmm1, zmm2, zmm3
// zmm1[i:i+31] += zmm2[i:i+7] * zmm3[i:i+7]     (4 个 u8×i8 乘积累加到 i32)
// 每个 32-bit lane 内做 4 个 8-bit 乘加
// 一条指令 = 64 个 INT8 乘加（16 lanes × 4 products/lane）
```

**吞吐量**：
- Cascade Lake: 2 FMA ports, 每周期 128 INT8 ops
- Ice Lake+: 同上，但有 AVX-512 IFMA 额外加速

**对 GEMM 的影响**：
- INT8 GEMM 吞吐量是 FP32 的 4 倍（每 lane 4 个乘加 vs 1 个 FMA）
- 需要 A 矩阵为 u8、B 矩阵为 i8（或反过来），结果累加到 i32
- 量化推理的核心指令

**Rust intrinsic**：
```rust
// 需要 target_feature = "avx512vnni"
_mm512_dpbusd_epi32(src, a, b)  // src += a(u8) * b(i8)
```

### 3.2 AVX-512 BF16

**指令**：`vdpbf16ps`

**核心操作**：
```
vdpbf16ps zmm1, zmm2, zmm3
// zmm1[i:i+31] += zmm2[i:i+15](bf16) * zmm3[i:i+15](bf16)  (高16bit)
//                + zmm2[i+16:i+31](bf16) * zmm3[i+16:i+31](bf16) (低16bit)
// 每个 32-bit lane 内做 2 个 bf16 乘加，累加到 f32
// 一条指令 = 32 个 BF16 乘加（16 lanes × 2 products/lane）
```

**吞吐量**：FP32 GEMM 的 2 倍（每 lane 2 个乘加 vs 1 个 FMA）

**Rust intrinsic**：
```rust
// 需要 target_feature = "avx512bf16"
_mm512_dpbf16_ps(src, a, b)  // src(f32) += a(bf16) * b(bf16)
```

### 3.3 Intel AMX (Advanced Matrix Extensions)

**硬件**：Sapphire Rapids+ (SPR, 4th Gen Xeon)

**架构**：
- 8 个 Tile 寄存器 (TMM0-TMM7)，每个 1KB (16 行 × 64 字节)
- TILECFG 寄存器配置 tile 尺寸
- 专用矩阵乘法单元，独立于 AVX-512 FMA

**核心指令**：
```
TDPBF16PS tmm1, tmm2, tmm3   // tmm1(f32) += tmm2(bf16) × tmm3(bf16)
TDPBSSD   tmm1, tmm2, tmm3   // tmm1(i32) += tmm2(i8)  × tmm3(i8)
TDPBSUD   tmm1, tmm2, tmm3   // tmm1(i32) += tmm2(i8)  × tmm3(u8)
TDPBUSD   tmm1, tmm2, tmm3   // tmm1(i32) += tmm2(u8)  × tmm3(i8)
TDPBUUD   tmm1, tmm2, tmm3   // tmm1(i32) += tmm2(u8)  × tmm3(u8)
TDPFP16PS tmm1, tmm2, tmm3   // tmm1(f32) += tmm2(fp16) × tmm3(fp16) (Granite Rapids)
```

**Tile 尺寸**（oneDNN 源码确认）：
- 最大 16 行 × 64 字节
- BF16: 16×32 元素（每元素 2 字节）
- INT8: 16×64 元素（每元素 1 字节）
- 结果 tile: 16×16 f32（每元素 4 字节，16 行 × 16 列）

**oneDNN 的 AMX 分块策略**（`brgemm_blocking_tmm`）：
```
bd_block (M方向) = max 16（tile 行数限制）
ld_block (N方向) = 16（固定，tile 列数）
ld_block2 = 2 或 3（多个 N tile 组合）
```

**吞吐量**：
- BF16: 每周期 16×32×16 = 8192 BF16 ops（vs AVX-512 BF16 的 32 ops/cycle）
- INT8: 每周期 16×64×16 = 16384 INT8 ops
- 约为 AVX-512 的 8-16 倍

**对 gllm-kernels 的影响**：
- AMX 需要完全不同的微内核结构（tile load → tile multiply → tile store）
- 不能复用 FMA 微内核的宏模板
- 需要单独的 `define_matmul_amx!` 宏
- Rust 目前没有稳定的 AMX intrinsic，需要 inline asm 或 nightly feature

### 3.4 指令吞吐量对比总结

| 指令 | 数据类型 | 每条指令 ops | 每周期 ops (2 FMA ports) | vs FP32 FMA |
|------|----------|-------------|------------------------|-------------|
| `vfmadd231ps zmm` | f32 | 16 FMA | 32 | 1× |
| `vdpbf16ps zmm` | bf16→f32 | 32 FMA | 64 | 2× |
| `vpdpbusd zmm` | u8×i8→i32 | 64 MAC | 128 | 4× |
| `TDPBF16PS tmm` | bf16→f32 | 8192 | 8192 | ~256× |
| `TDPBSSD tmm` | i8×i8→i32 | 16384 | 16384 | ~512× |

---

## 4. 微内核设计原则

### 4.1 寄存器分配策略

微内核的核心约束：**累加器数量 = MR × (NR/LANES)**，必须 ≤ 可用向量寄存器数。

| ISA | 总寄存器 | 累加器 | B 加载 | A broadcast | 可用 MR×NR |
|-----|---------|--------|--------|-------------|------------|
| AVX2 (16×ymm) | 16 | 12 | 2 | 1 | 6×16 (NV=2) |
| AVX-512 (32×zmm) | 32 | 28 | 2 | 1 | 14×32 (NV=2) |
| NEON (32×128b) | 32 | 24 | 4 | 4 | 8×12 (NV=3) |
| NEON M1 (32×128b) | 32 | 24 | 4 | 4 | 12×8 |

**当前 gllm-kernels 的选择**：
- AVX-512: TM=14, NV=2 → 14×2=28 累加器 + 2 B + 1 A broadcast = 31 ✅
- AVX2: TM=6, NV=2 → 6×2=12 累加器 + 2 B + 1 A broadcast = 15 ✅

### 4.2 K 维度展开 (K-unrolling)

当前实现使用 8× 展开（`ku = kc & !7`），这是正确的选择：
- 隐藏 FMA 延迟（4-5 周期）：8 次独立 FMA 足够填满流水线
- 减少循环开销
- 允许编译器更好地调度指令

BLIS 的 Haswell 微内核也使用 k=8 展开。

### 4.3 NEON 微内核的特殊性

NEON 与 x86 的关键差异：
1. **没有 broadcast 指令**：需要 `vld1q_dup_f32` 或 `vdupq_laneq_f32`
2. **128-bit 宽度**：每个向量只有 4 个 f32，需要更多向量覆盖 N 维度
3. **双发射 FMA**：M1 可以每周期 2 个 FMLA，但需要独立的源操作数
4. **NEON 没有 prefetch 指令的精确控制**：`__builtin_prefetch` 效果有限

BLIS Firestorm (M1) 选择 MR=12, NR=8：
- 12 行 × 8 列 / 4 lanes = 12×2 = 24 累加器
- 剩余 8 个寄存器用于 A/B 加载和临时值

### 4.4 SVE (Scalable Vector Extension)

SVE 的特殊性：向量长度在编译时未知（128-2048 bit），运行时确定。

**对微内核的影响**：
- 不能硬编码 LANES，需要 `svcntw()` 运行时查询
- 循环结构需要用 predicate 寄存器控制尾部处理
- MR 仍然可以编译时确定（与向量长度无关）
- NR = NV × svcntw()，NV 编译时确定

**BLIS A64FX (SVE 512-bit) 参数**：
- 预期 MR=10, NR=12（类似 AVX-512 的比例）

**对 gllm-kernels 的建议**：
- SVE 支持可以延后，当前 NEON 覆盖所有 ARM 场景
- 如果实现，需要新的 `define_matmul_sve!` 宏，不能复用 NEON 模板

---

## 5. 预取策略

### 5.1 当前实现的预取

```rust
simd_primitive!($isa, $elem, prefetch, bp.add(TN*16) as *const i8, 0);
```

预取距离 = TN×16 = 32×16 = 512 字节 (AVX-512) 或 16×16 = 256 字节 (AVX2)。
这在 K 循环的 8× 展开中，每 8 次迭代预取一次。

### 5.2 最优预取策略（基于 BLIS/oneDNN 实践）

**B 矩阵预取**（已 packed，顺序访问）：
- 预取距离 = 8-16 个 cache line ahead（512B-1KB）
- 当前实现的 TN×16 ≈ 512B 是合理的 ✅

**A 矩阵预取**（broadcast 访问，步长 = k_size）：
- 当前实现没有 A 预取 ⚠️
- 当 k_size > 1024 时，A 的行间距超过 L1，需要预取
- 建议：在 K 循环中添加 `prefetch(ac.add(k_size * $R + 64))`

**C 矩阵预取**（写回时）：
- 当前实现没有 C 预取
- 对于大 N，C 的行间距可能很大
- 建议：在 K 循环开始前预取 C tile 的目标地址

### 5.3 预取距离经验值

| ISA | B 预取距离 | A 预取距离 | 说明 |
|-----|-----------|-----------|------|
| AVX2 | 256-512B | 128-256B | L1=32KB, 较保守 |
| AVX-512 | 512-1024B | 256-512B | L1=48KB (ICL+), 可更激进 |
| NEON | 128-256B | 64-128B | L1=64KB (A57) / 128KB (M1) |

---

## 6. 流水线与编译器提示

### 6.1 FMA 延迟隐藏

FMA 指令延迟：
- Haswell/SKX: 4 周期延迟，2 port 吞吐
- M1: 4 周期延迟，4 FMLA/cycle (2 NEON pipes × 2)

要完全隐藏延迟，需要 `延迟 × 吞吐 = 4 × 2 = 8` 个独立 FMA 在飞行中。

当前实现的 8× K 展开 + MR 个独立累加器行，提供了足够的 ILP。

### 6.2 Rust 编译器提示

```rust
#[target_feature(enable = "avx512f,avx512bw")]  // 启用 ISA
#[inline(always)]                                 // 内联微内核
unsafe fn microkernel(...) { ... }

// 循环展开提示（Rust 没有 #pragma unroll，依赖编译器）
// 使用 const 泛型或手动展开（当前实现的 8× 手动展开是正确的）

// 分支预测提示
#[cold] fn handle_remainder(...) { ... }  // 标记冷路径
```

### 6.3 关键编译选项

```toml
[profile.release]
lto = "fat"           # 跨 crate 内联
codegen-units = 1     # 最大优化机会
panic = "abort"       # 消除 unwind 开销
```

```bash
RUSTFLAGS="-C target-cpu=native"  # 启用本机所有 ISA 特性
```

---

## 7. 对 gllm-kernels 的优化建议（优先级排序）

### P0: 立即可做的改进

1. **A 矩阵预取**：在 K 内循环中添加 A 的 prefetch，预期 5-10% 提升
2. **C 矩阵预取**：在 tile 计算开始前预取 C 的写回地址

### P1: 中期改进

3. **MC 分块**：为 Prefill (M>32) 场景添加 MC 维度分块，确保 A 的工作集装入 L2
4. **可选 Pack A**：当 M > 4×MR 且 K > 512 时启用 A packing
5. **VNNI INT8 微内核**：为 INT8 量化推理添加 `vpdpbusd` 路径，4× 吞吐提升
6. **BF16 微内核**：添加 `vdpbf16ps` 路径，2× 吞吐提升

### P2: 长期改进

7. **AMX 支持**：为 Sapphire Rapids+ 添加 tile 微内核，需要全新宏模板
8. **SVE 支持**：为 ARM SVE 添加可变宽度微内核
9. **多线程**：在 Loop 5 (NC) 或 Loop 3 (MC) 层级并行化

### P3: 可选改进

10. **NC 分块**：当 N > 4096 时添加 NC 维度分块（LLM 中 N 通常 ≤ 4096）
11. **动态 KC 调整**：根据运行时检测的 Cache 大小调整 KC

---

## 8. 参考文献与源码

| 来源 | 内容 | 链接 |
|------|------|------|
| Goto & van de Geijn, 2008 | "Anatomy of High-Performance Matrix Multiplication" | ACM TOMS |
| BLIS haswell config | AVX2 MR=6, NR=16, KC=256 | `flame/blis/config/haswell/` |
| BLIS skx config | AVX-512 MR=32, NR=12, KC=384 | `flame/blis/config/skx/` |
| BLIS firestorm config | M1 MR=12, NR=8, KC=4096 | `flame/blis/config/firestorm/` |
| BLIS cortexa57 config | NEON MR=8, NR=12, KC=640 | `flame/blis/config/cortexa57/` |
| oneDNN brgemm_utils.cpp | AMX tile blocking: bd=16, ld=16 | `oneapi-src/oneDNN/src/cpu/x64/brgemm/` |
| Intel ISA Reference | VNNI/BF16/AMX 指令规格 | Intel SDM Vol.2 |
