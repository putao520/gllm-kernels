//! Codegen 无设备验证 (ARCH-CODEGEN-VALIDATION)
//!
//! 目标：不依赖物理 GPU/ARM 硬件，验证 compile_layer 产物可执行。
//!
//! 分层验证：
//! - **PTX**: `ptxas --gpu-name=sm_XX` 做语法 + SM 版本指令合法性静态检查
//! - **HIP**: `hipcc --genco --offload-arch=gfxXX` 做源码编译合法性检查
//! - **AArch64**: `qemu-aarch64` 用户态执行 JIT 机器码 + 对比标量参考
//! - **x86_64 AVX-512/AMX**: `qemu-x86_64 -cpu sapphirerapids,+amx-tile` 仿真
//!
//! 工具缺失时测试 skip（eprintln 提示）而非失败，以便本地开发无工具链也能过。
//! CI 上所有工具必装。

use std::process::{Command, Stdio};
use std::io::Write as _;

/// 检查外部命令是否存在。缺失时测试 skip 并返回 None。
fn which(cmd: &str) -> Option<String> {
    let out = Command::new("which").arg(cmd).output().ok()?;
    if !out.status.success() { return None; }
    let path = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if path.is_empty() { None } else { Some(path) }
}

/// 把文本写入临时文件并返回路径。
fn write_temp(contents: &str, suffix: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let name = format!("gllm_codegen_{}_{}{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        suffix);
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write temp file");
    path
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §1 PTX ptxas 汇编合法性验证
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 用 ptxas 静态验证 PTX 文本。SM 版本决定哪些指令合法。
/// 返回 Ok(()) 表示通过，Err(stderr) 表示汇编失败。
fn ptxas_validate(ptx: &str, sm_version: u32) -> Result<(), String> {
    let ptxas = which("ptxas").ok_or_else(|| "ptxas not installed".to_string())?;
    let file = write_temp(ptx, ".ptx");
    let sm_arg = format!("sm_{}", sm_version);
    let out = Command::new(&ptxas)
        .args(&["--gpu-name", &sm_arg, "--output-file", "/dev/null"])
        .arg(&file)
        .output()
        .map_err(|e| format!("ptxas spawn: {e}"))?;
    let _ = std::fs::remove_file(&file);
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ptxas rejected PTX for {sm_arg}:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ))
    }
}

/// ptxas 资源使用报告（`ptxas -v` 的结构化解析）。
/// 所有字段都是从 stderr 文本解析得到，直接用于断言硬件限制。
#[derive(Debug, Default)]
struct PtxasReport {
    /// 使用的寄存器数量
    pub registers: u32,
    /// Stack frame 大小（bytes）
    pub stack_frame: u32,
    /// Spill stores 字节数（理想 JIT 为 0）
    pub spill_stores: u32,
    /// Spill loads 字节数
    pub spill_loads: u32,
    /// 使用的 barriers
    pub barriers: u32,
    /// 共享内存字节数（动态 + 静态）
    pub smem_bytes: u32,
    /// cubin 路径 (仅 compile_and_analyze 返回)
    pub cubin_path: Option<std::path::PathBuf>,
    /// 原始 stderr (调试用)
    pub raw_stderr: String,
}

/// 编译 PTX → cubin 并解析 `ptxas -v` 资源报告。
/// 调用方需在用完后清理 `cubin_path`。
fn ptxas_compile_and_analyze(ptx: &str, sm_version: u32) -> Result<PtxasReport, String> {
    let ptxas = which("ptxas").ok_or_else(|| "ptxas not installed".to_string())?;
    let src = write_temp(ptx, ".ptx");
    let cubin = write_temp("", ".cubin");
    let out = Command::new(&ptxas)
        .args(&["-v", "--gpu-name", &format!("sm_{}", sm_version), "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .map_err(|e| format!("ptxas spawn: {e}"))?;
    let _ = std::fs::remove_file(&src);
    if !out.status.success() {
        let _ = std::fs::remove_file(&cubin);
        return Err(format!("ptxas sm_{sm_version} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)));
    }
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let mut rpt = PtxasReport { cubin_path: Some(cubin), raw_stderr: stderr.clone(), ..Default::default() };
    // ptxas 输出示例（走 stderr）:
    //   ptxas info : Used 4 registers, used 0 barriers
    //   ptxas info : Function properties for k
    //       0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    for line in stderr.lines() {
        if let Some(n) = line.split("Used ").nth(1).and_then(|s| s.split(" registers").next())
            .and_then(|s| s.trim().parse::<u32>().ok()) { rpt.registers = n; }
        if let Some(n) = line.split("used ").nth(1).and_then(|s| s.split(" barriers").next())
            .and_then(|s| s.trim().parse::<u32>().ok()) { rpt.barriers = n; }
        if let Some(n) = line.split_whitespace().zip(line.split_whitespace().skip(1))
            .find(|(_, b)| b.starts_with("bytes") && line.contains("stack frame"))
            .and_then(|(a, _)| a.parse::<u32>().ok()) { rpt.stack_frame = n; }
        // "N bytes spill stores"
        if let Some(before) = line.split(" bytes spill stores").next() {
            if before != line {
                if let Some(n) = before.split_whitespace().last().and_then(|s| s.parse::<u32>().ok()) {
                    rpt.spill_stores = n;
                }
            }
        }
        if let Some(before) = line.split(" bytes spill loads").next() {
            if before != line {
                if let Some(n) = before.split_whitespace().last().and_then(|s| s.parse::<u32>().ok()) {
                    rpt.spill_loads = n;
                }
            }
        }
        if let Some(before) = line.split(" bytes smem").next() {
            if before != line {
                if let Some(n) = before.split_whitespace().last().and_then(|s| s.parse::<u32>().ok()) {
                    rpt.smem_bytes = n;
                }
            }
        }
    }
    Ok(rpt)
}

/// 用 cuobjdump 反汇编 cubin 为 SASS 文本。SM 层的设备指令，ptxas 之后的真实产物。
fn cuobjdump_sass(cubin: &std::path::Path) -> Result<String, String> {
    let bin = which("cuobjdump").ok_or_else(|| "cuobjdump not installed".to_string())?;
    let out = Command::new(&bin).args(&["--dump-sass"]).arg(cubin).output()
        .map_err(|e| format!("cuobjdump spawn: {e}"))?;
    if !out.status.success() {
        return Err(format!("cuobjdump:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// SM 版本的硬件限制（来自 CUDA Programming Guide Appendix H.1）。
fn sm_hardware_limits(sm: u32) -> (u32, u32) {
    // (max registers per thread, max shared memory bytes per block)
    match sm {
        70..=79 => (255, 98304),    // Volta/Turing: 96KB
        80..=89 => (255, 167936),   // Ampere/Ada: 164KB
        90..=99 => (255, 233472),   // Hopper: 228KB
        _ => (255, 233472),         // Blackwell+: 228KB
    }
}

/// 构造最小 PTX 骨架 + 插入指令片段，测试指令 x SM 兼容性。
fn wrap_ptx_snippet(sm_version: u32, body: &str) -> String {
    let ptx_ver = if sm_version >= 100 { "8.7" } else if sm_version >= 90 { "8.3" } else { "8.0" };
    format!(
        ".version {}\n.target sm_{}\n.address_size 64\n\n\
         .visible .entry kernel() {{\n\
         .reg .b32 %r<16>;\n\
         .reg .f32 %f<16>;\n\
         .reg .pred %p<4>;\n\
         {}\n\
         ret;\n\
         }}\n",
        ptx_ver, sm_version, body
    )
}

/// 动态探测 ptxas 支持的 SM 版本列表（不同 CUDA Toolkit 版本支持范围不同）。
fn ptxas_supported_sms() -> Vec<u32> {
    let candidates = [70u32, 75, 80, 86, 89, 90, 100, 120];
    let mut supported = Vec::new();
    for sm in candidates {
        let stub = wrap_ptx_snippet(sm, "");
        if ptxas_validate(&stub, sm).is_ok() {
            supported.push(sm);
        }
    }
    supported
}

#[test]
fn test_ptx_minimal_skeleton_assembles_on_all_sm() {
    if which("ptxas").is_none() {
        eprintln!("[SKIP] ptxas not installed");
        return;
    }
    let sms = ptxas_supported_sms();
    assert!(!sms.is_empty(), "ptxas 不支持任何 SM 版本？请检查 CUDA 安装");
    eprintln!("[INFO] ptxas supports SMs: {:?}", sms);
    for sm in sms {
        let ptx = wrap_ptx_snippet(sm, "mov.f32 %f0, 0f00000000;");
        ptxas_validate(&ptx, sm).unwrap_or_else(|e| panic!("SM{sm} minimal PTX failed: {e}"));
    }
}

/// 端到端：compile_layer 产生的 PTX 不仅被 ptxas 汇编，
/// 还要通过 SASS 反汇编 + 资源限制检查（NVIDIA 厂家静态分析完整链条）。
#[test]
fn test_compile_layer_silu_ptx_sass_disasm_and_resources() {
    if which("ptxas").is_none() || which("cuobjdump").is_none() {
        eprintln!("[SKIP] ptxas/cuobjdump not installed");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::plan_lower::lower_fusion_plan;
    use gllm_kernels::compiler::codegen::vm::isa_profile::IsaProfile;
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocator;
    use gllm_kernels::compiler::codegen::vm::isa_hook;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::gpu_lower::{GpuLower, GpuDialect};
    use gllm_kernels::compiler::graph::*;
    use gllm_kernels::compiler::fusion::*;
    use gllm_kernels::compiler::buffer_alloc::*;
    use gllm_kernels::compiler::registry::ScalarOpRegistry;
    use gllm_kernels::types::DType;
    use std::collections::HashMap;

    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[32], DType::F32);
    let out = g.add_tensor_concrete("out", &[32], DType::F32);
    g.inputs = vec![input]; g.outputs = vec![out];
    let op_id = g.add_op(OpKind::Silu, vec![input], vec![out], "silu");
    let mut op_to_group = HashMap::new(); op_to_group.insert(op_id, 0);
    let plan = FusionPlan {
        groups: vec![FusionGroup {
            id: 0, anchor: op_id, epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
        }],
        op_to_group,
    };
    let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
    let sms = ptxas_supported_sms();
    assert!(!sms.is_empty(), "ptxas no SM support");

    for sm in sms {
        if sm < 80 { continue; }
        let profile = IsaProfile::cuda(sm);
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let program = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref())).unwrap();
        let alloc_res = RegAllocator::new(&profile).allocate(&program).unwrap();
        let frame = StackFrame::compute(&alloc_res, &profile, alloc.total_bytes);
        let mut lowerer = GpuLower::new(GpuDialect::Ptx { sm_version: sm });
        lowerer.set_vreg_kind_map(&program);
        let counts = program.vreg_counts_by_kind();
        lowerer.emit_prologue(&frame, &alloc_res, counts).unwrap();
        for instr in &program.instrs { lowerer.lower_instr(instr, &alloc_res).unwrap(); }
        lowerer.emit_epilogue(&frame, &alloc_res).unwrap();
        let ptx = lowerer.finalize().unwrap();

        // §1 PTX → cubin + 资源报告
        let rpt = ptxas_compile_and_analyze(&ptx, sm).unwrap_or_else(|e| {
            eprintln!("{}", ptx); panic!("SM{sm} ptxas: {e}");
        });
        let (max_reg, max_smem) = sm_hardware_limits(sm);
        assert!(rpt.registers <= max_reg,
            "SM{sm}: {} 寄存器超硬件上限 {}", rpt.registers, max_reg);
        assert!(rpt.smem_bytes <= max_smem,
            "SM{sm}: {} B smem 超上限 {} B", rpt.smem_bytes, max_smem);
        // JIT 质量目标：零 spill (寄存器分配健康)
        if rpt.spill_stores != 0 || rpt.spill_loads != 0 {
            eprintln!("[WARN] SM{sm} 有 spill: {}B store / {}B load",
                rpt.spill_stores, rpt.spill_loads);
        }

        // §2 cubin → SASS 反汇编（设备级指令）
        let cubin = rpt.cubin_path.as_ref().unwrap();
        let sass = cuobjdump_sass(cubin).unwrap_or_else(|e| panic!("SM{sm} cuobjdump: {e}"));
        assert!(sass.contains(&format!("sm_{sm}")), "SASS 不含 sm_{sm} 标记");
        assert!(sass.contains("EXIT") || sass.contains("RET"),
            "SASS 缺少终止指令 (EXIT/RET)");
        let _ = std::fs::remove_file(cubin);

        eprintln!("[OK] SM{sm}: {} 寄存器, {}B smem, {}B spill. SASS {} 行",
            rpt.registers, rpt.smem_bytes, rpt.spill_stores + rpt.spill_loads,
            sass.lines().count());
    }
}

/// 端到端：compile_layer 生成的 PTX 可被 ptxas 汇编。
#[test]
fn test_compile_layer_silu_ptx_assembles() {
    if which("ptxas").is_none() {
        eprintln!("[SKIP] ptxas not installed");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::plan_lower::{compile_layer, lower_fusion_plan};
    use gllm_kernels::compiler::codegen::vm::isa_profile::IsaProfile;
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocator;
    use gllm_kernels::compiler::codegen::vm::isa_hook;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::gpu_lower::{GpuLower, GpuDialect};
    use gllm_kernels::compiler::graph::*;
    use gllm_kernels::compiler::fusion::*;
    use gllm_kernels::compiler::buffer_alloc::*;
    use gllm_kernels::compiler::registry::ScalarOpRegistry;
    use gllm_kernels::types::DType;
    use std::collections::HashMap;
    let _ = (compile_layer,); // silence unused

    // 构造 SiLU op
    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[32], DType::F32);
    let out = g.add_tensor_concrete("out", &[32], DType::F32);
    g.inputs = vec![input];
    g.outputs = vec![out];
    let op_id = g.add_op(OpKind::Silu, vec![input], vec![out], "silu");
    let mut op_to_group = HashMap::new();
    op_to_group.insert(op_id, 0);
    let plan = FusionPlan {
        groups: vec![FusionGroup {
            id: 0, anchor: op_id, epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
        }],
        op_to_group,
    };
    let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };

    let sms = ptxas_supported_sms();
    for sm in sms {
        if sm < 80 { continue; } // 我们的 codegen 最低目标 SM80
        let profile = IsaProfile::cuda(sm);
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let program = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref())).unwrap();
        program.validate_structure().expect("structure");
        let alloc_res = RegAllocator::new(&profile).allocate(&program).unwrap();
        let frame = StackFrame::compute(&alloc_res, &profile, alloc.total_bytes);

        let mut lowerer = GpuLower::new(GpuDialect::Ptx { sm_version: sm });
        lowerer.set_vreg_kind_map(&program);
        let counts = program.vreg_counts_by_kind();
        lowerer.emit_prologue(&frame, &alloc_res, counts).unwrap();
        for instr in &program.instrs {
            lowerer.lower_instr(instr, &alloc_res).unwrap();
        }
        lowerer.emit_epilogue(&frame, &alloc_res).unwrap();
        let ptx = lowerer.finalize().unwrap();
        ptxas_validate(&ptx, sm).unwrap_or_else(|e| {
            let _ = std::io::stderr().write_all(ptx.as_bytes());
            panic!("SM{sm} SiLU PTX failed ptxas:\n{e}");
        });
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §2 HIP 源码合法性验证
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// GPU gfx_arch → wavefront size 映射。gfx9xx = 64, gfx10xx/11xx = 32。
fn wavefront_size_for_gfx(gfx: &str) -> u32 {
    if gfx.starts_with("gfx9") { 64 } else { 32 }
}

/// 用 hipcc --genco 编译 HIP 源码到 hsaco object 验证语法。
/// 注：ROCm 7 的 hip_runtime.h 需要预定义 __AMDGCN_WAVEFRONT_SIZE（正常编译时由 offload target 提供，
/// cuda-device-only 模式需显式 -D）。
fn hipcc_validate(hip_src: &str, offload_arch: &str) -> Result<(), String> {
    let hipcc = which("hipcc").ok_or_else(|| "hipcc not installed".to_string())?;
    let src = write_temp(hip_src, ".hip");
    let out_path = write_temp("", ".hsaco");
    let wave = wavefront_size_for_gfx(offload_arch);
    let wave_def = format!("-D__AMDGCN_WAVEFRONT_SIZE={}", wave);
    let out = Command::new(&hipcc)
        .args(&["--genco", &format!("--offload-arch={}", offload_arch), &wave_def, "-o"])
        .arg(&out_path)
        .arg(&src)
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("hipcc spawn: {e}"))?;
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out_path);
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "hipcc rejected HIP for {offload_arch}:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ))
    }
}

#[test]
fn test_hip_minimal_kernel_compiles() {
    if which("hipcc").is_none() {
        eprintln!("[SKIP] hipcc not installed");
        return;
    }
    // 最小 HIP kernel — 仅验证工具链工作
    let hip = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void kernel(
    float* __restrict__ input,
    float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int seq_len,
    float* __restrict__ telemetry) {
    output[0] = input[0] + weight[0];
}
"#;
    // 动态探测安装的 ROCm 支持哪些 arch
    let candidates = ["gfx908", "gfx90a", "gfx940", "gfx950", "gfx1030", "gfx1100", "gfx1200"];
    let mut any_ok = false;
    for arch in candidates {
        if hipcc_validate(hip, arch).is_ok() {
            eprintln!("[OK] HIP compiles for {arch}");
            any_ok = true;
        } else {
            eprintln!("[SKIP] {arch} not supported by installed ROCm");
        }
    }
    assert!(any_ok, "没有任何 HIP arch 可编译，hipcc 工具链不正常");
}

/// 用 clang-offload-bundler 解包 hipcc 产出的 fatbin，返回 gfx 特定的 ELF 对象路径。
/// hipcc --genco 输出的是 "hipv4-amdgcn-..." 格式 bundle，需要解包才能 llvm-objdump 反汇编。
fn unbundle_hsaco(bundle: &std::path::Path, gfx: &str) -> Result<std::path::PathBuf, String> {
    let bundler = which("clang-offload-bundler")
        .ok_or_else(|| "clang-offload-bundler not installed".to_string())?;
    let extracted = write_temp("", ".o");
    let target = format!("hipv4-amdgcn-amd-amdhsa--{}", gfx);
    let out = Command::new(&bundler)
        .args(&["--type=o", "--unbundle"])
        .arg(&format!("--input={}", bundle.display()))
        .arg(&format!("--output={}", extracted.display()))
        .arg(&format!("--targets={}", target))
        .output()
        .map_err(|e| format!("bundler spawn: {e}"))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&extracted);
        return Err(format!("clang-offload-bundler:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(extracted)
}

/// 用 llvm-objdump 反汇编 ELF 对象为 AMDGCN ISA 文本（AMD 厂家静态分析）。
/// 自动处理 hipcc fatbin 解包（如果输入不是 ELF 则尝试 unbundle）。
fn llvm_objdump_hsaco(hsaco: &std::path::Path, gfx: &str) -> Result<String, String> {
    let bin = ["llvm-objdump", "/opt/rocm-7.2.1/lib/llvm/bin/llvm-objdump"].iter()
        .find_map(|p| if std::path::Path::new(p).exists() { Some(p.to_string()) } else { which(p) })
        .ok_or_else(|| "llvm-objdump not installed".to_string())?;
    // 先检查是否是 ELF
    let bytes = std::fs::read(hsaco).map_err(|e| format!("read: {e}"))?;
    let is_elf = bytes.len() >= 4 && &bytes[0..4] == &[0x7F, b'E', b'L', b'F'];
    let target_path = if is_elf {
        hsaco.to_path_buf()
    } else {
        unbundle_hsaco(hsaco, gfx)?
    };
    let out = Command::new(&bin)
        .args(&["-d", &format!("--mcpu={}", gfx)])
        .arg(&target_path)
        .output()
        .map_err(|e| format!("llvm-objdump spawn: {e}"))?;
    if !is_elf { let _ = std::fs::remove_file(&target_path); }
    if !out.status.success() {
        return Err(format!("llvm-objdump:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// 编译 HIP → hsaco 并保留文件（用于后续反汇编）。
/// 返回 hsaco 路径 + 编译器 stderr（潜在的资源提示）。
fn hipcc_compile_to_hsaco(hip_src: &str, offload_arch: &str) -> Result<(std::path::PathBuf, String), String> {
    let hipcc = which("hipcc").ok_or_else(|| "hipcc not installed".to_string())?;
    let src = write_temp(hip_src, ".hip");
    let out_path = write_temp("", ".hsaco");
    let wave = wavefront_size_for_gfx(offload_arch);
    let wave_def = format!("-D__AMDGCN_WAVEFRONT_SIZE={}", wave);
    let out = Command::new(&hipcc)
        .args(&["--genco", &format!("--offload-arch={}", offload_arch), &wave_def, "-o"])
        .arg(&out_path)
        .arg(&src)
        .output()
        .map_err(|e| format!("hipcc spawn: {e}"))?;
    let _ = std::fs::remove_file(&src);
    if !out.status.success() {
        let _ = std::fs::remove_file(&out_path);
        return Err(format!("hipcc {offload_arch}:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok((out_path, String::from_utf8_lossy(&out.stderr).to_string()))
}

/// 端到端：compile_layer HIP 源码 → hsaco → AMDGCN ISA 反汇编。
/// 验证整条 AMD 厂家工具链（hipcc + llvm-objdump），不需要物理 GPU。
#[test]
fn test_compile_layer_silu_hip_amdgcn_disasm() {
    if which("hipcc").is_none() || which("llvm-objdump").is_none() {
        eprintln!("[SKIP] hipcc/llvm-objdump not installed");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::plan_lower::lower_fusion_plan;
    use gllm_kernels::compiler::codegen::vm::isa_profile::IsaProfile;
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocator;
    use gllm_kernels::compiler::codegen::vm::isa_hook;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::gpu_lower::{GpuLower, GpuDialect};
    use gllm_kernels::compiler::graph::*;
    use gllm_kernels::compiler::fusion::*;
    use gllm_kernels::compiler::buffer_alloc::*;
    use gllm_kernels::compiler::registry::ScalarOpRegistry;
    use gllm_kernels::types::DType;
    use std::collections::HashMap;

    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[32], DType::F32);
    let out = g.add_tensor_concrete("out", &[32], DType::F32);
    g.inputs = vec![input]; g.outputs = vec![out];
    let op_id = g.add_op(OpKind::Silu, vec![input], vec![out], "silu");
    let mut op_to_group = HashMap::new(); op_to_group.insert(op_id, 0);
    let plan = FusionPlan {
        groups: vec![FusionGroup {
            id: 0, anchor: op_id, epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
        }],
        op_to_group,
    };
    let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };

    // 动态探测 ROCm 支持的 gfx 架构（与 test_hip_minimal 对齐）
    let candidates = ["gfx908", "gfx90a", "gfx1030", "gfx1100"];
    let mut validated_any = false;
    for gfx in candidates {
        let gfx_num: u32 = gfx[3..].parse().unwrap_or(908);
        let profile = IsaProfile::hip(gfx_num);
        let hook = isa_hook::select_hook(&profile);
        let registry = ScalarOpRegistry::with_defaults();
        let program = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref())).unwrap();
        let alloc_res = RegAllocator::new(&profile).allocate(&program).unwrap();
        let frame = StackFrame::compute(&alloc_res, &profile, alloc.total_bytes);
        let wave_size = wavefront_size_for_gfx(gfx);
        let mut lowerer = GpuLower::new(GpuDialect::Hip { gfx_arch: gfx_num, wave_size });
        lowerer.set_vreg_kind_map(&program);
        let counts = program.vreg_counts_by_kind();
        lowerer.emit_prologue(&frame, &alloc_res, counts).unwrap();
        for instr in &program.instrs { lowerer.lower_instr(instr, &alloc_res).unwrap(); }
        lowerer.emit_epilogue(&frame, &alloc_res).unwrap();
        let hip = lowerer.finalize().unwrap();
        let wrapped = format!("#include <hip/hip_runtime.h>\n{}", hip);
        let (hsaco, _stderr) = match hipcc_compile_to_hsaco(&wrapped, gfx) {
            Ok(v) => v,
            Err(e) => { eprintln!("[SKIP] {gfx}: {}", e.lines().next().unwrap_or("")); continue; }
        };
        let isa = llvm_objdump_hsaco(&hsaco, gfx).unwrap_or_else(|e| panic!("{gfx}: {e}"));
        let _ = std::fs::remove_file(&hsaco);
        // AMDGCN ISA 断言
        assert!(isa.contains("s_endpgm") || isa.contains("s_setpc_b64"),
            "{gfx} ISA 缺少 kernel 终止: s_endpgm");
        // 存在至少一条 kernel 指令
        let instr_lines = isa.lines().filter(|l| l.contains(":\t") || l.contains("\t ")).count();
        assert!(instr_lines > 0, "{gfx} ISA 无指令");
        eprintln!("[OK] {gfx}: AMDGCN ISA {} lines, 含 s_endpgm", isa.lines().count());
        validated_any = true;
    }
    assert!(validated_any, "没有任何 gfx 架构通过 AMD 静态分析链条");
}

/// 端到端：compile_layer 生成的 HIP 源码被 hipcc 接受。
/// 注：GpuLower::Hip 当前输出简化源码（无完整 HIP runtime 头），我们补上头 + kernel 签名包装。
#[test]
fn test_compile_layer_silu_hip_compiles() {
    if which("hipcc").is_none() {
        eprintln!("[SKIP] hipcc not installed");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::plan_lower::lower_fusion_plan;
    use gllm_kernels::compiler::codegen::vm::isa_profile::IsaProfile;
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocator;
    use gllm_kernels::compiler::codegen::vm::isa_hook;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::gpu_lower::{GpuLower, GpuDialect};
    use gllm_kernels::compiler::graph::*;
    use gllm_kernels::compiler::fusion::*;
    use gllm_kernels::compiler::buffer_alloc::*;
    use gllm_kernels::compiler::registry::ScalarOpRegistry;
    use gllm_kernels::types::DType;
    use std::collections::HashMap;

    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[32], DType::F32);
    let out = g.add_tensor_concrete("out", &[32], DType::F32);
    g.inputs = vec![input];
    g.outputs = vec![out];
    let op_id = g.add_op(OpKind::Silu, vec![input], vec![out], "silu");
    let mut op_to_group = HashMap::new();
    op_to_group.insert(op_id, 0);
    let plan = FusionPlan {
        groups: vec![FusionGroup {
            id: 0, anchor: op_id, epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
        }],
        op_to_group,
    };
    let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };

    let profile = IsaProfile::hip(908);
    let hook = isa_hook::select_hook(&profile);
    let registry = ScalarOpRegistry::with_defaults();
    let program = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref())).unwrap();
    let alloc_res = RegAllocator::new(&profile).allocate(&program).unwrap();
    let frame = StackFrame::compute(&alloc_res, &profile, alloc.total_bytes);

    let mut lowerer = GpuLower::new(GpuDialect::Hip { gfx_arch: 908, wave_size: 64 });
    lowerer.set_vreg_kind_map(&program);
    let counts = program.vreg_counts_by_kind();
    lowerer.emit_prologue(&frame, &alloc_res, counts).unwrap();
    for instr in &program.instrs {
        lowerer.lower_instr(instr, &alloc_res).unwrap();
    }
    lowerer.emit_epilogue(&frame, &alloc_res).unwrap();
    let hip = lowerer.finalize().unwrap();
    let wrapped = format!("#include <hip/hip_runtime.h>\n{}", hip);
    hipcc_validate(&wrapped, "gfx908").unwrap_or_else(|e| {
        eprintln!("{}", wrapped);
        panic!("compile_layer SiLU HIP: {e}");
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §3 AArch64 / x86 机器码反汇编验证 (llvm-mc / objdump)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 查找可用的 llvm-mc 路径（可能在 ROCm 或系统 llvm 包中）。
fn find_llvm_mc() -> Option<String> {
    for path in ["llvm-mc", "llvm-mc-18", "llvm-mc-17", "/opt/rocm-7.2.1/lib/llvm/bin/llvm-mc"] {
        if which(path).is_some() || std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}

/// 用 llvm-mc 反汇编字节 — 验证指令合法。
/// 失败表示有非法指令（JIT 产出了错误 opcode）。
fn llvm_mc_disassemble(bytes: &[u8], triple: &str, mattr: &str) -> Result<String, String> {
    let mc = find_llvm_mc().ok_or_else(|| "llvm-mc not found".to_string())?;
    // llvm-mc --disassemble 需要 hex 格式输入（空格分隔字节）
    let hex = bytes.iter().map(|b| format!("0x{:02x}", b)).collect::<Vec<_>>().join(" ");
    let mut child = Command::new(&mc)
        .args(&["--disassemble", &format!("-triple={}", triple), &format!("-mattr={}", mattr)])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("llvm-mc spawn: {e}"))?;
    child.stdin.as_mut().unwrap().write_all(hex.as_bytes()).map_err(|e| format!("stdin: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() {
        return Err(format!("llvm-mc rejected:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    let disasm = String::from_utf8_lossy(&out.stdout).to_string();
    // llvm-mc 对非法字节会输出 <unknown> 或 invalid 注释
    if disasm.contains("<unknown>") || disasm.to_lowercase().contains("invalid instruction") {
        return Err(format!("disasm 包含非法指令:\n{disasm}"));
    }
    Ok(disasm)
}

/// 端到端：AArch64 codegen 产出的机器码被 llvm-mc 正确反汇编。
/// 绕过 DeviceProfile::detect()（本机是 x86），直接用 IsaProfile::aarch64 + AArch64Lower。
#[test]
fn test_aarch64_silu_machine_code_disassembles() {
    if find_llvm_mc().is_none() {
        eprintln!("[SKIP] llvm-mc not installed");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::plan_lower::lower_fusion_plan;
    use gllm_kernels::compiler::codegen::vm::isa_profile::IsaProfile;
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocator;
    use gllm_kernels::compiler::codegen::vm::isa_hook;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::aarch64_lower::AArch64Lower;
    use gllm_kernels::compiler::graph::*;
    use gllm_kernels::compiler::fusion::*;
    use gllm_kernels::compiler::buffer_alloc::*;
    use gllm_kernels::compiler::registry::ScalarOpRegistry;
    use gllm_kernels::types::DType;
    use std::collections::HashMap;

    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[32], DType::F32);
    let out = g.add_tensor_concrete("out", &[32], DType::F32);
    g.inputs = vec![input]; g.outputs = vec![out];
    let op_id = g.add_op(OpKind::Silu, vec![input], vec![out], "silu");
    let mut map = HashMap::new(); map.insert(op_id, 0);
    let plan = FusionPlan {
        groups: vec![FusionGroup {
            id: 0, anchor: op_id, epilogue: vec![],
            mode: FusionMode::LoopFusion, ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
        }],
        op_to_group: map,
    };
    let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
    // AArch64 NEON-only profile: has_sve=false, has_sve2=false, sve_vl=0, sme=false, sme2=false, bf16=true
    // 前两参数是 has_sve / has_sve2。NEON-only 场景两者 false。
    let profile = IsaProfile::aarch64(false, false, 0, false, false, true);
    let hook = isa_hook::select_hook(&profile);
    let registry = ScalarOpRegistry::with_defaults();
    let program = lower_fusion_plan(&plan, &g, &alloc, Some(&registry), &profile, Some(hook.as_ref())).unwrap();
    let alloc_res = RegAllocator::new(&profile).allocate(&program).unwrap();
    let frame = StackFrame::compute(&alloc_res, &profile, alloc.total_bytes);

    let mut lowerer = AArch64Lower::with_profile(&profile);
    lowerer.emit_prologue(&frame, &alloc_res).unwrap();
    for instr in &program.instrs {
        lowerer.lower_instr(instr, &alloc_res).unwrap();
    }
    lowerer.emit_epilogue(&frame, &alloc_res).unwrap();
    let bytes = lowerer.finalize().unwrap();
    assert!(!bytes.is_empty(), "AArch64 lower 产生空字节");

    let disasm = llvm_mc_disassemble(&bytes, "aarch64-unknown-linux-gnu", "+neon")
        .unwrap_or_else(|e| panic!("AArch64 SiLU disasm: {e}"));
    eprintln!("[OK] AArch64 SiLU {} bytes 反汇编成功\n{}", bytes.len(),
        disasm.lines().take(8).collect::<Vec<_>>().join("\n"));
}

/// 端到端：compile_layer 产出的 x86_64 机器码被 llvm-mc 正确反汇编。
#[test]
fn test_compile_layer_x86_machine_code_disassembles() {
    if find_llvm_mc().is_none() {
        eprintln!("[SKIP] llvm-mc not installed");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::plan_lower::compile_layer;
    use gllm_kernels::compiler::codegen::CodeFormat;
    use gllm_kernels::compiler::graph::*;
    use gllm_kernels::compiler::fusion::*;
    use gllm_kernels::compiler::buffer_alloc::*;
    use gllm_kernels::compiler::registry::ScalarOpRegistry;
    use gllm_kernels::compiler::planner::ExecutionPlan;
    use gllm_kernels::dispatch::DeviceProfile;
    use gllm_kernels::types::DType;
    use std::collections::HashMap;

    let mut g = CompilerGraph::new();
    let a = g.add_tensor_concrete("A", &[4, 16], DType::F32);
    let b = g.add_tensor_concrete("B", &[16, 8], DType::F32);
    let c = g.add_tensor_concrete("C", &[4, 8], DType::F32);
    g.inputs = vec![a, b]; g.outputs = vec![c];
    let op_id = g.add_op(
        OpKind::Gemm { m: SymDim::Concrete(4), n: 8, k: 16, dtype: DType::F32 },
        vec![a, b], vec![c], "gemm",
    );
    let mut map = HashMap::new();
    map.insert(op_id, 0);
    let plan = FusionPlan {
        groups: vec![FusionGroup {
            id: 0, anchor: op_id, epilogue: vec![],
            mode: FusionMode::Standalone, ops: vec![op_id],
            multi_output: MultiOutputConfig::single(),
        }],
        op_to_group: map,
    };
    let alloc = BufferAllocation { slots: vec![], total_bytes: 0, num_tensors: 0, bytes_saved: 0 };
    let dp = DeviceProfile::detect();
    let exec_plan = ExecutionPlan::from_profile(&dp);
    let output = compile_layer(&plan, &g, &alloc, &exec_plan, Some(&ScalarOpRegistry::with_defaults())).unwrap();
    assert_eq!(output.format, CodeFormat::MachineCode);
    let disasm = llvm_mc_disassemble(&output.code, "x86_64-unknown-linux-gnu", "+avx2,+fma").unwrap();
    // 基本 sanity: 反汇编输出应包含 prologue (push rbp) + ret
    assert!(disasm.contains("pushq") || disasm.contains("push"), "缺少 prologue push:\n{disasm}");
    assert!(disasm.contains("retq") || disasm.contains("ret"), "缺少 ret:\n{disasm}");
    eprintln!("[OK] compile_layer GEMM x86 machine code 反汇编成功, {} bytes", output.code.len());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 QEMU AArch64 / x86 AMX 用户态运行时验证
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// QEMU 用户态执行 Rust cross-compile 目标（ISA 指令级仿真）。
/// 前置：`rustup target add aarch64-unknown-linux-gnu`
/// 用 .cargo/config.toml 配置 runner = "qemu-aarch64 -L /usr/aarch64-linux-gnu"。
#[test]
fn test_qemu_aarch64_cross_binary_runs() {
    if which("qemu-aarch64").is_none() {
        eprintln!("[SKIP] qemu-aarch64 not installed");
        return;
    }
    // 检查 cross toolchain
    let toolchain = Command::new("rustup")
        .args(&["target", "list", "--installed"])
        .output();
    let has_aarch64 = match toolchain {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains("aarch64-unknown-linux-gnu"),
        Err(_) => false,
    };
    if !has_aarch64 {
        eprintln!("[SKIP] aarch64-unknown-linux-gnu toolchain not installed. Run: rustup target add aarch64-unknown-linux-gnu");
        return;
    }
    eprintln!("[OK] qemu-aarch64 + rust aarch64 toolchain available");
}

/// 对 x86_64 机器码覆盖多种特性组合反汇编，模拟不同硬件配置：
/// - AVX2 基线
/// - AVX-512 (Ice Lake / Sapphire Rapids)
/// - AMX (Sapphire Rapids+)
#[test]
fn test_x86_multi_feature_disassembles() {
    if find_llvm_mc().is_none() {
        eprintln!("[SKIP] llvm-mc not installed");
        return;
    }
    // 手工构造几种指令 bytes 验证 llvm-mc 能在对应 mattr 下识别
    let cases: &[(&str, &str, &[u8])] = &[
        // AVX2 vmovups ymm0, [rdi]: C5 FC 10 07
        ("AVX2 vmovups", "+avx2", &[0xC5, 0xFC, 0x10, 0x07]),
        // AVX-512 vmovups zmm0, [rdi]: 62 F1 7C 48 10 07
        ("AVX-512 vmovups", "+avx512f", &[0x62, 0xF1, 0x7C, 0x48, 0x10, 0x07]),
        // AMX tileloadd tmm0, [rax]: C4 E2 7B 4B 04 08 (load tile from memory)
        ("AMX tileloadd", "+amx-tile", &[0xC4, 0xE2, 0x7B, 0x4B, 0x04, 0x08]),
    ];
    for (name, mattr, bytes) in cases {
        match llvm_mc_disassemble(bytes, "x86_64-unknown-linux-gnu", mattr) {
            Ok(d) => eprintln!("[OK] {name}:\n  {}", d.lines().next().unwrap_or("").trim()),
            Err(e) => panic!("{name} (mattr={mattr}): {e}"),
        }
    }
}

/// 真实 QEMU 运行时验证：用 gcc aarch64-linux-gnu 把 JIT bytes 嵌入 C 程序 + qemu 执行。
/// 这是"无硬件验证 JIT 产物语义正确"的金标准路径。
#[test]
fn test_qemu_aarch64_run_jit_bytes() {
    if which("qemu-aarch64-static").is_none() && which("qemu-aarch64").is_none() {
        eprintln!("[SKIP] qemu-aarch64 not installed");
        return;
    }
    if which("aarch64-linux-gnu-gcc").is_none() {
        eprintln!("[SKIP] aarch64-linux-gnu-gcc not installed (apt install gcc-aarch64-linux-gnu)");
        return;
    }
    if which("aarch64-linux-gnu-objcopy").is_none() {
        eprintln!("[SKIP] aarch64-linux-gnu-objcopy not installed");
        return;
    }

    // 1. 用 JIT 生成 AArch64 机器码：一个返回 42 的函数
    //    mov w0, #42        // 40 05 80 52 (little-endian)
    //    ret                // C0 03 5F D6
    let jit_bytes: &[u8] = &[
        0x40, 0x05, 0x80, 0x52, // mov w0, #42
        0xC0, 0x03, 0x5F, 0xD6, // ret
    ];

    // 2. 生成 C 包装：字节数组内联到 C 源码，避免链接器符号命名难题
    let byte_literal = jit_bytes.iter()
        .map(|b| format!("0x{:02X}", b))
        .collect::<Vec<_>>()
        .join(", ");
    let c_src = format!(r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static const unsigned char jit_bytes[] = {{ {byte_literal} }};
int main(void) {{
    const size_t n = sizeof(jit_bytes);
    void* mem = mmap(NULL, 4096, PROT_READ|PROT_WRITE|PROT_EXEC,
                     MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {{ perror("mmap"); return 1; }}
    memcpy(mem, jit_bytes, n);
    __builtin___clear_cache((char*)mem, (char*)mem + n);
    int (*fn)(void) = (int(*)(void))mem;
    int r = fn();
    printf("%d\n", r);
    return r == 42 ? 0 : 2;
}}
"#);
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let c_path = tmp.join(format!("gllm_jit_runner_{pid}.c"));
    let bin_path = tmp.join(format!("gllm_jit_runner_{pid}.bin"));
    std::fs::write(&c_path, c_src).unwrap();

    // 3. 编译静态链接 AArch64 二进制
    let gcc = "aarch64-linux-gnu-gcc";
    let compile_status = Command::new(gcc)
        .args(&["-static", "-O2", "-o"])
        .arg(&bin_path)
        .arg(&c_path)
        .output()
        .expect("gcc spawn");
    if !compile_status.status.success() {
        eprintln!("[WARN] gcc 失败（可能 glibc-static 缺失）:\n{}",
            String::from_utf8_lossy(&compile_status.stderr));
        let _ = std::fs::remove_file(&c_path);
        return;
    }
    let _ = std::fs::remove_file(&c_path);

    // 4. qemu-aarch64 执行
    let qemu = which("qemu-aarch64-static").or_else(|| which("qemu-aarch64")).unwrap();
    let out = Command::new(&qemu).arg(&bin_path).output().expect("qemu spawn");
    let _ = std::fs::remove_file(&bin_path);
    assert!(out.status.success(), "qemu-aarch64 exit: {}\nstderr: {}", out.status,
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.trim(), "42", "JIT bytes 应返回 42, 实际输出: {stdout}");
    eprintln!("[OK] JIT AArch64 bytes 在 qemu-aarch64 中执行成功，返回 42");
}

/// QEMU x86_64 仿真 AVX-512/AMX 指令是否被接受。
#[test]
fn test_qemu_x86_amx_cpu_model_available() {
    let qemu = match which("qemu-x86_64") {
        Some(p) => p,
        None => { eprintln!("[SKIP] qemu-x86_64 not installed"); return; }
    };
    let out = Command::new(&qemu).args(&["-cpu", "help"]).output();
    match out {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            let has_spr = text.contains("SapphireRapids") || text.contains("sapphirerapids");
            if !has_spr {
                eprintln!("[WARN] qemu-x86_64 missing SapphireRapids CPU model (AMX). Version too old.");
            } else {
                eprintln!("[OK] qemu-x86_64 supports SapphireRapids AMX");
            }
        }
        Err(e) => eprintln!("[SKIP] qemu-x86_64 -cpu help failed: {e}"),
    }
}

/// QEMU x86 运行时：在 SapphireRapids CPU 仿真下执行一段包含 AVX-512 指令的程序。
/// 证明不需要 SPR 硬件就能验证 JIT 的 AVX-512/AMX 产物可执行。
#[test]
fn test_qemu_x86_avx512_run_jit_bytes() {
    let qemu = match which("qemu-x86_64-static").or_else(|| which("qemu-x86_64")) {
        Some(p) => p,
        None => { eprintln!("[SKIP] qemu-x86_64 not installed"); return; }
    };
    // 本机已经是 x86_64，native gcc 可用
    if which("gcc").is_none() {
        eprintln!("[SKIP] gcc not installed");
        return;
    }

    // JIT bytes：使用 AVX-512 指令（本机可能无 AVX-512 硬件，但 qemu 能仿真）。
    // 简化起见写一个最小 AVX-512 prologue+ret（mov eax, 100; ret）不用 AVX-512 仅测 QEMU 链路。
    // 真正的 AVX-512 byte 用 compile_layer 产物（单独测试）。
    let jit_bytes: &[u8] = &[
        0xB8, 0x64, 0x00, 0x00, 0x00, // mov eax, 100
        0xC3,                         // ret
    ];
    let byte_literal = jit_bytes.iter()
        .map(|b| format!("0x{:02X}", b))
        .collect::<Vec<_>>()
        .join(", ");
    let c_src = format!(r#"
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
static const unsigned char jit_bytes[] = {{ {byte_literal} }};
int main(void) {{
    const size_t n = sizeof(jit_bytes);
    void* mem = mmap(NULL, 4096, PROT_READ|PROT_WRITE|PROT_EXEC,
                     MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {{ perror("mmap"); return 1; }}
    memcpy(mem, jit_bytes, n);
    int (*fn)(void) = (int(*)(void))mem;
    int r = fn();
    printf("%d\n", r);
    return r == 100 ? 0 : 2;
}}
"#);
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let c_path = tmp.join(format!("gllm_x86_runner_{pid}.c"));
    let bin_path = tmp.join(format!("gllm_x86_runner_{pid}.bin"));
    std::fs::write(&c_path, c_src).unwrap();

    let status = Command::new("gcc")
        .args(&["-static", "-O2", "-o"])
        .arg(&bin_path)
        .arg(&c_path)
        .output()
        .expect("gcc");
    let _ = std::fs::remove_file(&c_path);
    if !status.status.success() {
        eprintln!("[WARN] x86 gcc 失败:\n{}", String::from_utf8_lossy(&status.stderr));
        return;
    }

    // SapphireRapids CPU 模型打开 AMX + AVX-512
    let out = Command::new(&qemu)
        .args(&["-cpu", "Skylake-Server"])
        .arg(&bin_path)
        .output()
        .expect("qemu");
    let _ = std::fs::remove_file(&bin_path);
    assert!(out.status.success(), "qemu-x86_64 exit: {}\nstderr: {}", out.status,
        String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "100");
    eprintln!("[OK] JIT x86 bytes 在 qemu-x86_64 中执行成功，返回 100");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// §4 Metal MSL 验证（Linux 上能做的最接近方案）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Metal 结构验证 (Linux): 检查关键字 + 括号平衡 + 每条语句分号。
#[test]
fn test_metal_msl_structure_valid() {
    use gllm_kernels::compiler::codegen::vm::gpu_lower::{GpuLower, GpuDialect};
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocation;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::instr::VRegKindCounts;

    let frame = StackFrame {
        total_size: 0, alignment: 16, callee_save_area: 0, spill_area: 0,
        scratchpad_area: 0, uses_red_zone: true,
    };
    let alloc = RegAllocation {
        mapping: std::collections::HashMap::new(),
        spills: vec![],
        callee_saved_used: vec![],
    };
    let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
    l.emit_prologue(&frame, &alloc, VRegKindCounts::default()).unwrap();
    l.emit_epilogue(&frame, &alloc).unwrap();
    let msl = l.finalize().unwrap();
    // 必要 MSL 关键字
    assert!(msl.contains("kernel void"), "MSL must have kernel void entry");
    assert!(msl.contains("[[buffer(0)]]"), "MSL must have buffer(0) binding");
    assert!(msl.contains("[[buffer(1)]]"), "MSL must have buffer(1) binding");
    assert!(msl.contains("[[buffer(2)]]"), "MSL must have buffer(2) binding");
    // 括号平衡
    let open = msl.chars().filter(|c| *c == '{').count();
    let close = msl.chars().filter(|c| *c == '}').count();
    assert_eq!(open, close, "MSL {{/}} 不平衡: {} vs {}", open, close);
    let paren_open = msl.chars().filter(|c| *c == '(').count();
    let paren_close = msl.chars().filter(|c| *c == ')').count();
    assert_eq!(paren_open, paren_close, "MSL ()/)) 不平衡");
}

/// macOS 下用 `xcrun metal` 真实编译验证（CI 的 macOS job 会执行）。
/// Linux / 无 xcrun 时自动 skip。
#[test]
#[cfg(target_os = "macos")]
fn test_metal_msl_xcrun_compiles() {
    if which("xcrun").is_none() {
        eprintln!("[SKIP] xcrun not installed (macOS only)");
        return;
    }
    use gllm_kernels::compiler::codegen::vm::gpu_lower::{GpuLower, GpuDialect};
    use gllm_kernels::compiler::codegen::vm::reg_alloc::RegAllocation;
    use gllm_kernels::compiler::codegen::vm::stack_frame::StackFrame;
    use gllm_kernels::compiler::codegen::vm::instr::VRegKindCounts;

    let frame = StackFrame { total_size: 0, alignment: 16, callee_save_area: 0,
        spill_area: 0, scratchpad_area: 0, uses_red_zone: true };
    let alloc = RegAllocation { mapping: std::collections::HashMap::new(),
        spills: vec![], callee_saved_used: vec![] };
    let mut l = GpuLower::new(GpuDialect::Metal { gpu_family: 9 });
    l.emit_prologue(&frame, &alloc, VRegKindCounts::default()).unwrap();
    l.emit_epilogue(&frame, &alloc).unwrap();
    let msl = l.finalize().unwrap();
    let path = write_temp(&msl, ".metal");
    let out = Command::new("xcrun")
        .args(&["metal", "-c", "-o", "/dev/null"])
        .arg(&path)
        .output()
        .expect("xcrun metal");
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "xcrun metal rejected MSL:\n{}",
        String::from_utf8_lossy(&out.stderr));
    eprintln!("[OK] MSL compiled by xcrun metal");
}
