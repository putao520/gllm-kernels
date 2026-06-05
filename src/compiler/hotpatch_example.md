# Hot JMP Patching — 使用示例

## 概述

Hot JMP Patching（§9.2）允许在运行时原子修改 JIT 生成代码中的跳转指令，实现零停机的拓扑重构。

## 核心组件

### 1. HotPatchRegistry

管理所有可热修补的跳转点：

```rust
use gllm_kernels::compiler::HotPatchRegistry;

// 创建注册表（code_base 指向 JIT 生成的可执行内存）
let registry = HotPatchRegistry::new(code_base, code_len);
```

### 2. 在 JIT Codegen 中预留跳转点

在 x86_64 codegen 中使用 `emit_patchable_jmp`：

```rust
// 生成可热修补的 JMP 指令
let patch_id = codegen.emit_patchable_jmp(default_path_label)?;

// 添加可选目标
codegen.add_patch_alternative(patch_id, optimized_path_addr)?;
codegen.add_patch_alternative(patch_id, fallback_path_addr)?;
```

### 3. 运行时热修补

```rust
// 原子修改跳转目标
unsafe {
    registry.patch(patch_id, new_target_addr)?;
}
```

## 使用场景

### 场景 1: MoE 专家动态切换

```rust
// 初始：跳转到所有专家
let patch_id = codegen.emit_patchable_jmp(all_experts_label)?;

// 运行时检测到专家 3 长期零命中
// → 热修补跳过专家 3
unsafe {
    registry.patch(patch_id, skip_expert_3_addr)?;
}
```

### 场景 2: Attention 策略切换

```rust
// 初始：使用 FlashAttention
let patch_id = codegen.emit_patchable_jmp(flash_attn_label)?;

// 检测到序列长度超过阈值
// → 切换到 Paged Attention
unsafe {
    registry.patch(patch_id, paged_attn_addr)?;
}
```

### 场景 3: Early Exit 动态启用

```rust
// 初始：执行完整 32 层
let patch_id = codegen.emit_patchable_jmp(layer_32_label)?;

// 检测到输出概率收敛
// → 在第 16 层提前退出
unsafe {
    registry.patch(patch_id, early_exit_16_addr)?;
}
```

## 架构约束

### 1. 内存保护

JIT 代码通常是只读+可执行（W^X）。`HotPatchRegistry::patch` 内部会临时修改内存保护：

```rust
// Linux: mprotect(PROT_READ | PROT_WRITE | PROT_EXEC)
// 修改完成后恢复只读+可执行
```

### 2. 原子性

使用 `AtomicU64` 存储当前目标地址，确保并发安全：

```rust
pub struct PatchableJump {
    pub offset: usize,
    pub target: AtomicU64,  // 原子操作
    pub alternatives: Vec<u64>,
}
```

### 3. 指令格式

x86_64 JMP rel32 指令格式：

```
E9 [4-byte signed offset]
```

相对偏移计算：`target - (jmp_addr + 5)`

## 与 JIT Director Daemon 集成

SPEC §9.2 要求后台 JIT Director Daemon 监控 telemetry 数据，触发热修补：

```rust
// 伪代码
loop {
    // 扫描 KV Page Headers 中的 telemetry 数据
    let stats = scan_telemetry_buffer();
    
    // 检测全局共识不可逆突变
    if stats.expert_hit_rate[3] < 0.001 && stats.duration > 1_000_000 {
        // 在沙盒中编译新路径
        let new_code = compile_skip_expert_3()?;
        
        // 原子热修补
        unsafe {
            registry.patch(expert_patch_id, new_code.entry_addr)?;
        }
    }
    
    sleep(Duration::from_secs(10));
}
```

## 测试

```bash
# 运行 hotpatch 单元测试
cargo test --lib compiler::hotpatch

# 验证所有测试通过
cargo test --lib
```

## 限制

1. **仅支持 x86_64**：AArch64 需要不同的指令格式（B/BR）
2. **rel32 范围**：跳转目标必须在 ±2GB 范围内
3. **单线程修改**：同一时刻只能有一个线程修改跳转点

## 未来扩展

- [ ] AArch64 支持（B 指令，26-bit offset）
- [ ] GPU PTX 支持（BRA 指令）
- [ ] 多路径跳转表（switch-case 优化）
- [ ] 热修补历史记录（用于回滚）
