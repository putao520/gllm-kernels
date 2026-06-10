impl GpuLower {
    pub fn new(dialect: GpuDialect) -> Self {
        use crate::dispatch::device_profile::DeviceProfile;
        let isa_profile = super::isa_profile::IsaProfile::from_device_profile(&DeviceProfile::detect());
        Self {
            dialect,
            ir: String::new(),
            indent: 0,
            loop_label_counter: 0,
            loop_stack: Vec::new(),
            skip_label_counter: 0,
            tmem_allocated: false,
            abi_param_names: vec!["input_ptr", "weight_ptr", "output_ptr", "seq_len", "telemetry_ptr"],
            scratch_vec_names: vec!["%fs0", "%fs1", "%fs2"],
            scratch_gpr_names: vec!["%rs0", "%rs1", "%rs_bound"],
            scratch_pred_names: vec!["%ps0", "%ps1"],
            vreg_kinds: Vec::new(),
            path_labels: std::collections::HashMap::new(),
            epilogue_label: "EPILOGUE".to_string(),
            path_label_counter: 0,
            jit_ctx: crate::compiler::jit_context::JitContext::new(&isa_profile),
        }
    }

    /// 从 VmProgram 提取 VReg → Kind 映射 (ARCH-GPU-REG-KIND)。
    /// 必须在 lower_instr 之前调用，否则 reg_name 无法为未分配的 VReg 选择正确命名空间。
    pub fn set_vreg_kind_map(&mut self, program: &VmProgram) {
        let count = program.vreg_count() as usize;
        self.vreg_kinds = vec![None; count];
        for instr in &program.instrs {
            if let VmInstr::DeclareVReg { id, kind, .. } = instr {
                let idx = id.0 as usize;
                if idx < self.vreg_kinds.len() {
                    self.vreg_kinds[idx] = Some(*kind);
                }
            }
        }
    }

    /// ARCH-GPU-REG-NAME: 根据 VReg 的 Kind + dialect 选择命名空间。
    /// PTX: %r/%rd/%f/%p/%t（PTX 虚拟寄存器语法）
    /// HIP/Metal: 合法 C++ 标识符 (r_N/rd_N/f_N/p_N/t_N)
    fn reg_name_with_kind(&self, vreg: VRegId, alloc: &RegAllocation) -> String {
        let (prefix_gpr32, prefix_gpr64, prefix_vec, prefix_mask, prefix_tile) = match self.dialect {
            GpuDialect::Ptx { .. } => ("%r", "%rd", "%f", "%p", "%t"),
            GpuDialect::Hip { .. } | GpuDialect::Metal { .. } => ("r_", "rd_", "f_", "p_", "t_"),
        };
        if let Some(phys) = alloc.get(vreg) {
            return match phys {
                PhysReg::Gpr(g) => format!("{prefix_gpr32}{}", g.0),
                PhysReg::Vec(v) => format!("{prefix_vec}{}", v.0),
                PhysReg::Tile(t) => format!("{prefix_tile}{}", t.0),
                PhysReg::Mask(m) => format!("{prefix_mask}{}", m.0),
                // GPU 后端不经线性扫描 RegAllocator 的 CPU spill 路径；如果 RegAllocator
                // 在 GPU profile 下 spill 了，这是配置 bug (GPU 通常寄存器数远大于 VReg 数)。
                PhysReg::Spilled(slot) => panic!(
                    "GPU codegen does not support spilled VReg (slot={} v{}); \
                     GPU RegAllocator 必须确保每个 VReg 都能分到物理寄存器",
                    slot, vreg.0,
                ),
            };
        }
        let kind = self.vreg_kinds.get(vreg.0 as usize).copied().flatten();
        match kind {
            Some(VRegKind::Ptr) => format!("{prefix_gpr64}{}", vreg.0),
            Some(VRegKind::Scalar) | Some(VRegKind::Counter) | Some(VRegKind::ByteOffset) => format!("{prefix_gpr32}{}", vreg.0),
            Some(VRegKind::Vec) => format!("{prefix_vec}{}", vreg.0),
            Some(VRegKind::Mask) => format!("{prefix_mask}{}", vreg.0),
            Some(VRegKind::Tile) => format!("{prefix_tile}{}", vreg.0),
            None => format!("{prefix_vec}{}", vreg.0),
        }
    }

    fn param_name(&self, idx: u8) -> Result<&'static str, CompilerError> {
        self.abi_param_names.get(idx as usize).copied()
            .ok_or_else(|| CompilerError::CodegenViolation(format!(
                "GPU ABI param idx {} 超出范围 [0..{})", idx, self.abi_param_names.len())))
    }

    fn emit_line(&mut self, line: &str) {
        for _ in 0..self.indent { self.ir.push_str("  "); }
        self.ir.push_str(line);
        self.ir.push('\n');
    }

    fn next_loop_label(&mut self) -> u32 {
        let id = self.loop_label_counter;
        self.loop_label_counter += 1;
        id
    }

    fn next_skip_label(&mut self) -> u32 {
        let id = self.skip_label_counter;
        self.skip_label_counter += 1;
        id
    }

    fn sm_version(&self) -> Option<u32> {
        match self.dialect { GpuDialect::Ptx { sm_version } => Some(sm_version), _ => None }
    }

    fn gfx_arch(&self) -> Option<u32> {
        match self.dialect { GpuDialect::Hip { gfx_arch, .. } => Some(gfx_arch), _ => None }
    }

    // ── Utility ──

    fn offset_to_string(&self, offset: &OffsetExpr, alloc: &RegAllocation) -> String {
        match offset {
            OffsetExpr::Const(c) => format!("{c}"),
            OffsetExpr::LoopOffset(v) => self.reg_name_with_kind(*v, alloc),
            OffsetExpr::Add(a, b) => format!("({}+{})", self.offset_to_string(a, alloc), self.offset_to_string(b, alloc)),
            OffsetExpr::Mul(inner, scale) => format!("({}*{scale})", self.offset_to_string(inner, alloc)),
            OffsetExpr::ScalarVReg(v) => self.reg_name_with_kind(*v, alloc),
        }
    }

    /// Resolve barrier_name to PTX mbarrier symbol.
    ///
    /// Prologue declares `.shared .align 8 .b64 mbar[4];` for SM90+.
    /// barrier_name maps to indices: "ping" → 0, "pong" → 1, "gate" → 2, others → hash % 4.
    fn resolve_barrier_symbol(&self, barrier_name: &str) -> String {
        let idx: u32 = match barrier_name {
            "ping" => 0,
            "pong" => 1,
            "gate" => 2,
            _ => {
                // Simple hash-based mapping for custom names
                let hash = barrier_name.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
                hash % 4
            }
        };
        format!("mbar+{idx}")
    }

    // ── Prologue / Epilogue ──

    fn emit_shared_mem_decl(&mut self, size_bytes: usize) {
        match self.dialect {
            GpuDialect::Ptx { .. } => self.emit_line(&format!(".shared .align 16 .b8 smem[{size_bytes}];")),
            GpuDialect::Hip { .. } => self.emit_line(&format!("__shared__ float smem[{}];", size_bytes / 4)),
            GpuDialect::Metal { .. } => self.emit_line(&format!("threadgroup float smem[{}];", size_bytes / 4)),
        }
    }

    /// HIP/Metal (C++ 风格) 下为每个 VRegKind 声明 C++ 变量。
    /// reg_name_with_kind 会产生 f_0/f_1/rd_0/r_0/p_0 这样的标识符，必须先声明。
    fn emit_hip_cxx_var_decls(&mut self, counts: VRegKindCounts) {
        let upper = counts.gpr_like()
            .max(counts.vec_like())
            .max(counts.mask_like())
            .max(counts.tile_like());
        for i in 0..upper {
            // 每个可能的 VReg.0 都声明三种身份（rd_/r_/f_/p_），
            // 多余的变量被编译器 DCE — 保证下游代码任意引用都合法。
            self.emit_line(&format!("float f_{i} = 0.0f;"));
            self.emit_line(&format!("unsigned int r_{i} = 0;"));
            self.emit_line(&format!("float* rd_{i} = nullptr;"));
            self.emit_line(&format!("bool p_{i} = false;"));
        }
        // Lower scratch names (与 PTX scratch_vec_names 镜像)
        self.emit_line("float fs_0 = 0.0f, fs_1 = 0.0f, fs_2 = 0.0f;");
        self.emit_line("unsigned int rs_0 = 0, rs_1 = 0, rs_bound = 0;");
        self.emit_line("bool ps_0 = false, ps_1 = false;");
        self.emit_line("float hip_shfl_tmp = 0.0f;");
    }

    fn emit_reg_decl(&mut self, counts: VRegKindCounts) {
        if let GpuDialect::Ptx { .. } = self.dialect {
            // ARCH-GPU-REG-NAMESPACE: vec_like/gpr_like/etc 已经 = max_id + 1
            // 因为 VReg 全局编号，我们必须为每个命名空间声明 [0, max_VReg_id] 的上界，
            // 即使中间许多编号给别的 kind 用——PTX 允许声明的槽位未使用。
            // 更稳妥：用 VmProgram 的 next_vreg 作为共同上界（所有命名空间声明相同容量）。
            // 这会稍多浪费但保证任意 VRegId 作为 %f/%r/%p 都有声明。
            let global_upper = counts.gpr_like()
                .max(counts.vec_like())
                .max(counts.mask_like())
                .max(counts.tile_like());
            let vec_count = global_upper.max(1);
            let gpr_count = global_upper.max(1);
            let mask_count = global_upper.max(16);
            let tile_count = counts.tile_like().max(1);
            self.emit_line(&format!(".reg .f32 %f<{vec_count}>;"));
            self.emit_line(&format!(".reg .b32 %r<{gpr_count}>;"));
            self.emit_line(&format!(".reg .b64 %rd<{gpr_count}>;"));
            self.emit_line(&format!(".reg .pred %p<{mask_count}>;"));
            if tile_count > 1 {
                self.emit_line(&format!(".reg .b32 %t<{tile_count}>;"));
            }
            // ARCH-GPU-SCRATCH: Lower 独占 scratch 独立命名空间
            let vec_names: Vec<&str> = self.scratch_vec_names.clone();
            let gpr_names: Vec<&str> = self.scratch_gpr_names.clone();
            let pred_names: Vec<&str> = self.scratch_pred_names.clone();
            for name in vec_names {
                self.emit_line(&format!(".reg .f32 {name};"));
            }
            for name in gpr_names {
                self.emit_line(&format!(".reg .b32 {name};"));
            }
            for name in pred_names {
                self.emit_line(&format!(".reg .pred {name};"));
            }
            // TMEM 地址寄存器 (SM100+)
            self.emit_line(".reg .b32 %tmem_addr;");
            self.emit_line(".reg .b32 %smem_tmem_addr;");
            // ARCH-GPU-PTX-ADDR: VecLoad/VecStore 用 64-bit 地址 scratch
            self.emit_line(".reg .b64 %rd_addr;");
        }
    }

    /// ARCH-GPU-REG-COUNT: 接受 VReg 计数动态生成寄存器声明。
    /// 调用方传入 VmProgram::vreg_counts_by_kind() 结果。
    pub fn emit_prologue(
        &mut self,
        frame: &StackFrame,
        _alloc: &RegAllocation,
        vreg_counts: VRegKindCounts,
    ) -> Result<(), CompilerError> {
        match self.dialect {
            GpuDialect::Ptx { sm_version } => {
                let ptx_ver = if sm_version >= 100 { "8.7" } else if sm_version >= 90 { "8.3" } else if sm_version >= 70 { "8.0" } else if sm_version >= 60 { "6.5" } else { "5.0" };
                self.emit_line(&format!(".version {ptx_ver}"));
                self.emit_line(&format!(".target sm_{sm_version}"));
                self.emit_line(".address_size 64");
                self.emit_line("");
                self.emit_line(".visible .entry kernel(");
                self.emit_line("  .param .u64 input_ptr,");
                self.emit_line("  .param .u64 weight_ptr,");
                self.emit_line("  .param .u64 output_ptr,");
                self.emit_line("  .param .u32 seq_len,");
                self.emit_line("  .param .u64 telemetry_ptr");
                self.emit_line(") {");
                self.indent += 1;
                self.emit_reg_decl(vreg_counts);
                if frame.scratchpad_area > 0 {
                    self.emit_shared_mem_decl(frame.scratchpad_area);
                }
                // SM90+: mbarrier 声明
                if sm_version >= 90 {
                    self.emit_line(".shared .align 8 .b64 mbar[4];");
                }
            }
            GpuDialect::Hip { gfx_arch, .. } => {
                // ARCH-GPU-PARAM-NAMES: 与 abi_param_names 严格一致（input_ptr/weight_ptr/...）
                self.emit_line(&format!("// HIP kernel (gfx{gfx_arch})"));
                self.emit_line("extern \"C\" __global__ void kernel(");
                self.emit_line("  float* __restrict__ input_ptr,");
                self.emit_line("  float* __restrict__ weight_ptr,");
                self.emit_line("  float* __restrict__ output_ptr,");
                self.emit_line("  unsigned int seq_len,");
                self.emit_line("  float* __restrict__ telemetry_ptr");
                self.emit_line(") {");
                self.indent += 1;
                // HIP 是 C++，需要为 VReg 声明变量
                self.emit_hip_cxx_var_decls(vreg_counts);
                if frame.scratchpad_area > 0 {
                    self.emit_shared_mem_decl(frame.scratchpad_area);
                }
            }
            GpuDialect::Metal { .. } => {
                self.emit_line("kernel void kernel_fn(");
                self.emit_line("  device float* input_ptr [[buffer(0)]],");
                self.emit_line("  device float* weight_ptr [[buffer(1)]],");
                self.emit_line("  device float* output_ptr [[buffer(2)]],");
                self.emit_line("  constant uint& seq_len [[buffer(3)]],");
                self.emit_line("  device float* telemetry_ptr [[buffer(4)]]");
                self.emit_line(") {");
                self.indent += 1;
                // Metal 也是 C++ like
                self.emit_hip_cxx_var_decls(vreg_counts);
                if frame.scratchpad_area > 0 {
                    // Metal: threadgroup float smem[N] — PLE 的 scratchpad
                    self.emit_shared_mem_decl(frame.scratchpad_area);
                }
            }
        }
        Ok(())
    }

    /// Emit mega-kernel PTX prologue with full 21-parameter ABI.
    pub fn emit_mega_kernel_prologue(
        &mut self,
        frame: &StackFrame,
        _alloc: &RegAllocation,
        vreg_counts: VRegKindCounts,
    ) -> Result<(), CompilerError> {
        match self.dialect {
            GpuDialect::Ptx { sm_version } => {
                let ptx_ver = if sm_version >= 100 { "8.7" } else if sm_version >= 90 { "8.3" } else if sm_version >= 70 { "8.0" } else if sm_version >= 60 { "6.5" } else { "5.0" };
                self.emit_line(&format!(".version {ptx_ver}"));
                self.emit_line(&format!(".target sm_{sm_version}"));
                self.emit_line(".address_size 64");
                self.emit_line("");
                self.emit_line(".visible .entry mega_kernel(");
                // 20 ABI parameters matching the mega-kernel ABI
                self.emit_line("  .param .u64 input_ids_ptr,");
                self.emit_line("  .param .u64 weight_blob_ptr,");
                self.emit_line("  .param .u64 kv_cache_ptr,");
                self.emit_line("  .param .u64 positions_ptr,");
                self.emit_line("  .param .u64 aux_ptr,");
                self.emit_line("  .param .u32 batch_size,");
                self.emit_line("  .param .u32 seq_len,");
                self.emit_line("  .param .u64 scratchpad_ptr,");
                self.emit_line("  .param .u64 output_tokens_ptr,");
                self.emit_line("  .param .u32 temperature_bits,");
                self.emit_line("  .param .u32 top_k,");
                self.emit_line("  .param .u32 top_p_bits,");
                self.emit_line("  .param .u32 max_new_tokens,");
                self.emit_line("  .param .u32 eos_token_id,");
                self.emit_line("  .param .u64 hook_ctx_ptr,");
                self.emit_line("  .param .u64 telemetry_ptr,");
                self.emit_line("  .param .u32 session_position,");
                self.emit_line("  .param .u64 fused_hidden_ptr,");
                self.emit_line("  .param .u32 num_mm_tokens,");
                self.emit_line("  .param .u64 callback_table_ptr");
                self.emit_line(") {");
                self.indent += 1;
                // Load parameters into registers
                self.emit_line("ld.param.u64 %rd_input, [input_ids_ptr];");
                self.emit_line("ld.param.u64 %rd_weights, [weight_blob_ptr];");
                self.emit_line("ld.param.u64 %rd_kvcache, [kv_cache_ptr];");
                self.emit_line("ld.param.u64 %rd_positions, [positions_ptr];");
                self.emit_line("ld.param.u64 %rd_aux, [aux_ptr];");
                self.emit_line("ld.param.u32 %r_batch, [batch_size];");
                self.emit_line("ld.param.u32 %r_seq, [seq_len];");
                self.emit_line("ld.param.u64 %rd_scratch, [scratchpad_ptr];");
                self.emit_line("ld.param.u64 %rd_output, [output_tokens_ptr];");
                self.emit_line("ld.param.u32 %r_temp, [temperature_bits];");
                self.emit_line("ld.param.u32 %r_topk, [top_k];");
                self.emit_line("ld.param.u32 %r_topp, [top_p_bits];");
                self.emit_line("ld.param.u32 %r_maxnew, [max_new_tokens];");
                self.emit_line("ld.param.u32 %r_eos, [eos_token_id];");
                self.emit_line("ld.param.u64 %rd_hook, [hook_ctx_ptr];");
                self.emit_line("ld.param.u64 %rd_telem, [telemetry_ptr];");
                self.emit_line("ld.param.u32 %r_session, [session_position];");
                self.emit_line("ld.param.u64 %rd_fused, [fused_hidden_ptr];");
                self.emit_line("ld.param.u32 %r_nmm, [num_mm_tokens];");
                self.emit_line("ld.param.u64 %rd_cb, [callback_table_ptr];");
                self.emit_reg_decl(vreg_counts);
                if frame.scratchpad_area > 0 {
                    self.emit_shared_mem_decl(frame.scratchpad_area);
                }
                if sm_version >= 90 {
                    self.emit_line(".shared .align 8 .b64 mbar[4];");
                }
            }
            GpuDialect::Hip { gfx_arch, .. } => {
                self.emit_line(&format!("// HIP mega-kernel (gfx{gfx_arch})"));
                self.emit_line("extern \"C\" __global__ void mega_kernel(");
                self.emit_line("  unsigned int* __restrict__ input_ids_ptr,");
                self.emit_line("  unsigned char* __restrict__ weight_blob_ptr,");
                self.emit_line("  unsigned char* __restrict__ kv_cache_ptr,");
                self.emit_line("  unsigned int* __restrict__ positions_ptr,");
                self.emit_line("  unsigned int* __restrict__ aux_ptr,");
                self.emit_line("  unsigned int batch_size,");
                self.emit_line("  unsigned int seq_len,");
                self.emit_line("  unsigned char* __restrict__ scratchpad_ptr,");
                self.emit_line("  unsigned int* __restrict__ output_tokens_ptr,");
                self.emit_line("  unsigned int temperature_bits,");
                self.emit_line("  unsigned int top_k,");
                self.emit_line("  unsigned int top_p_bits,");
                self.emit_line("  unsigned int max_new_tokens,");
                self.emit_line("  unsigned int eos_token_id,");
                self.emit_line("  unsigned char* __restrict__ hook_ctx_ptr,");
                self.emit_line("  unsigned char* __restrict__ telemetry_ptr,");
                self.emit_line("  unsigned int session_position,");
                self.emit_line("  unsigned char* __restrict__ fused_hidden_ptr,");
                self.emit_line("  unsigned int num_mm_tokens,");
                self.emit_line("  unsigned char* __restrict__ callback_table_ptr");
                self.emit_line(") {");
                self.indent += 1;
                self.emit_hip_cxx_var_decls(vreg_counts);
                if frame.scratchpad_area > 0 {
                    self.emit_shared_mem_decl(frame.scratchpad_area);
                }
            }
            GpuDialect::Metal { .. } => {
                // Metal mega-kernel with full ABI
                self.emit_line("kernel void mega_kernel(");
                self.emit_line("  device unsigned int* input_ids_ptr [[buffer(0)]],");
                self.emit_line("  device unsigned char* weight_blob_ptr [[buffer(1)]],");
                self.emit_line("  device unsigned char* kv_cache_ptr [[buffer(2)]],");
                self.emit_line("  device unsigned int* positions_ptr [[buffer(3)]],");
                self.emit_line("  device unsigned int* aux_ptr [[buffer(4)]],");
                self.emit_line("  constant unsigned int& batch_size [[buffer(5)]],");
                self.emit_line("  constant unsigned int& seq_len [[buffer(6)]],");
                self.emit_line("  device unsigned char* scratchpad_ptr [[buffer(7)]],");
                self.emit_line("  device unsigned int* output_tokens_ptr [[buffer(8)]],");
                self.emit_line("  constant unsigned int& temperature_bits [[buffer(9)]],");
                self.emit_line("  constant unsigned int& top_k [[buffer(10)]],");
                self.emit_line("  constant unsigned int& top_p_bits [[buffer(11)]],");
                self.emit_line("  constant unsigned int& max_new_tokens [[buffer(12)]],");
                self.emit_line("  constant unsigned int& eos_token_id [[buffer(13)]],");
                self.emit_line("  device unsigned char* hook_ctx_ptr [[buffer(14)]],");
                self.emit_line("  device unsigned char* telemetry_ptr [[buffer(15)]],");
                self.emit_line("  constant unsigned int& session_position [[buffer(16)]],");
                self.emit_line("  device unsigned char* fused_hidden_ptr [[buffer(17)]],");
                self.emit_line("  constant unsigned int& num_mm_tokens [[buffer(18)]],");
                self.emit_line("  device unsigned char* callback_table_ptr [[buffer(19)]]");
                self.emit_line(") {");
                self.indent += 1;
                self.emit_hip_cxx_var_decls(vreg_counts);
                if frame.scratchpad_area > 0 {
                    self.emit_shared_mem_decl(frame.scratchpad_area);
                }
            }
        }
        // Update abi_param_names for mega-kernel ABI
        self.abi_param_names = vec![
            "input_ids_ptr", "weight_blob_ptr", "kv_cache_ptr", "positions_ptr",
            "aux_ptr", "batch_size", "seq_len", "scratchpad_ptr", "output_tokens_ptr",
            "temperature_bits", "top_k", "top_p_bits", "max_new_tokens", "eos_token_id",
            "hook_ctx_ptr", "telemetry_ptr", "session_position",
            "fused_hidden_ptr", "num_mm_tokens", "callback_table_ptr",
        ];
        Ok(())
    }

    pub fn emit_epilogue(&mut self, _frame: &StackFrame, _alloc: &RegAllocation) -> Result<(), CompilerError> {
        // SM100+: TMEM 释放
        if self.tmem_allocated {
            self.emit_line("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tmem_addr, 32;");
            self.tmem_allocated = false;
        }
        self.indent = self.indent.saturating_sub(1);
        self.emit_line("}");
        Ok(())
    }

    // ── 指令降低 ──
}

