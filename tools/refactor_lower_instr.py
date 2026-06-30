#!/usr/bin/env python3
"""
BCE-20260630-LOWER-INSTR-GOD-MATCH P2: aarch64 + gpu lower_instr 三级拆分。

机械抽取 (行为保持, 不改逻辑):
  - 解析 lower_instr 的巨型 match instr { arm1 => {body1}, arm2 => {body2}, ... }
  - 按 VmInstr::category() 把 arms 分到 8 类 (Memory/Arith/Control/Tile/Quant/GpuComm/Sampling/Misc)
  - 生成 L0 dispatch (lower_instr_inner: match instr.category() => lower_<cat>_<isa>)
  - 生成 L1 变体路由 (lower_<cat>_<isa>: match instr { VmInstr::X{..} => self.lower_<variant>_<isa>(instr, alloc), ... })
  - 生成 L2 叶子 fn (lower_<variant>_<isa>: match instr { VmInstr::X{..} => {<原 arm body>} _ => Err(...) })

用法: python3 tools/refactor_lower_instr.py <isa> <lower_instr.inc.rs路径>
  isa = aarch64 | gpu
"""
import re
import sys
import os

# ── 分类器 (镜像 vm_instr_category.rs, 保持单一真相) ──
CATEGORY_MAP = {
    # Memory
    'VecLoad': 'Memory', 'VecStore': 'Memory', 'VecNarrow': 'Memory', 'VecWiden': 'Memory',
    'Mov': 'Memory', 'Broadcast': 'Memory', 'LoadPtr': 'Memory', 'ScalarLoad': 'Memory',
    'ScalarStore': 'Memory', 'VecScalarStore': 'Memory', 'ScalarToIndex': 'Memory',
    'IndexToScalar': 'Memory', 'IntMulStride': 'Memory', 'GatherLoad': 'Memory',
    'ScatterStore': 'Memory', 'TableLookup': 'Memory', 'ScalarByteLoad': 'Memory',
    'MemCopy': 'Memory', 'AddPtr': 'Memory', 'StoreConstToStack': 'Memory',
    'MemFence': 'Memory', 'AtomicAdd': 'Memory', 'AtomicCAS': 'Memory',
    'ActivationSwap': 'Memory', 'Prefetch': 'Memory',
    # Arith
    'VecBinOp': 'Arith', 'VecShiftImm': 'Arith', 'VecUnaryOp': 'Arith', 'VecCmp': 'Arith',
    'VecCast': 'Arith', 'ConditionalSelect': 'Arith', 'Fma': 'Arith', 'HReduce': 'Arith',
    'Accumulate': 'Arith', 'Transcendental': 'Arith', 'DotProduct': 'Arith',
    'ScaleApply': 'Arith', 'GprBinOp': 'Arith', 'GprUnaryOp': 'Arith', 'GprLoadImm': 'Arith',
    'VecShuffle': 'Arith', 'VecExtractLane': 'Arith', 'VecInsertLane': 'Arith',
    'VecLoadConst': 'Arith',
    # Control
    'LoopBegin': 'Control', 'LoopEnd': 'Control', 'ScopeBegin': 'Control',
    'ScopeEnd': 'Control', 'ConditionalSkip': 'Control', 'GprCondAction': 'Control',
    'IndirectJump': 'Control', 'ConditionalExit': 'Control', 'BranchIfPtrNonNull': 'Control',
    'BranchIfGprZero': 'Control', 'BranchIfGprLtU': 'Control', 'UnconditionalBranch': 'Control',
    'BreakLoop': 'Control', 'MarkLabel': 'Control',
    # Tile
    'TileConfig': 'Tile', 'TileLoad': 'Tile', 'TileMma': 'Tile', 'TileStore': 'Tile',
    'TileRelease': 'Tile', 'TmemAlloc': 'Tile', 'TmemLoad': 'Tile', 'TmemStore': 'Tile',
    'TmemDealloc': 'Tile',
    # Quant
    'KiviQuantChannel': 'Quant', 'KiviQuantToken': 'Quant', 'KiviDequantLoad': 'Quant',
    'GgufSubScaleLoad': 'Quant', 'GgufKQuantScaleLoad': 'Quant', 'QuantBroadcastInt': 'Quant',
    'QuantScalarCvtLoad': 'Quant', 'QuantBlockLoad': 'Quant', 'QuantBiPlaneLoad': 'Quant',
    'QuantLoadBytesVec': 'Quant', 'QuantCodebookLookup': 'Quant', 'QuantExtractBits': 'Quant',
    'QuantDequantFma': 'Quant', 'QuantInterleave': 'Quant', 'QuantConcatSeq': 'Quant',
    'Q3KDecodeStep': 'Quant', 'HwQuantDequant': 'Quant', 'BitwiseGemm': 'Quant',
    'SparseGemm': 'Quant', 'SparseFp8Gemm': 'Quant', 'NativeFp4Gemm': 'Quant',
    'NativeFp8Gemm': 'Quant', 'SparseMaskIntersect': 'Quant',
    # GpuComm (无条件)
    'WarpSync': 'GpuComm', 'AsyncCopy': 'GpuComm', 'AsyncWait': 'GpuComm',
    'SharedMemAlloc': 'GpuComm', 'SharedMemStore': 'GpuComm', 'SharedMemLoad': 'GpuComm',
    'SharedMemAsyncStore': 'GpuComm', 'SharedMemAsyncWaitGroup': 'GpuComm',
    'WeightPrefetchAsync': 'GpuComm', 'WeightPrefetchWait': 'GpuComm',
    'WarpRoleDeclare': 'GpuComm', 'WarpBarrierArrive': 'GpuComm', 'WarpBarrierWait': 'GpuComm',
    'TmaDescriptorInit': 'GpuComm', 'Tma2DCopy': 'GpuComm', 'BarrierInit': 'GpuComm',
    'BlockSync': 'GpuComm', 'WarpReduce': 'GpuComm', 'SharedMemSwizzle': 'GpuComm',
    'ClusterBarrierInit': 'GpuComm', 'ClusterStore': 'GpuComm', 'ClusterLoad': 'GpuComm',
    'PageTableAddr': 'GpuComm', 'PageTableKVWrite': 'GpuComm',
    'PageTableKVWriteQuant': 'GpuComm',
    # GpuComm (nccl feature-gated)
    'AllReduceChunk': 'GpuComm', 'CommBarrier': 'GpuComm', 'NvlinkAsyncCopy': 'GpuComm',
    'RemotePageLookup': 'GpuComm', 'P2pPageFetch': 'GpuComm', 'RdmaPageFetch': 'GpuComm',
    'RdmaPageFetchCompressed': 'GpuComm', 'RemotePageAttn': 'GpuComm',
    'PageMigrationLock': 'GpuComm', 'PageMigrationUnlock': 'GpuComm',
    'PageLocationUpdate': 'GpuComm',
    # Sampling
    'Argmax': 'Sampling', 'TemperatureScale': 'Sampling', 'StoreToken': 'Sampling',
    'CheckStopCondition': 'Sampling', 'SoftmaxReduceMax': 'Sampling',
    'SoftmaxExpSum': 'Sampling', 'SoftmaxNormalize': 'Sampling',
    'SampleTopKFilter': 'Sampling', 'SampleTopPFilter': 'Sampling',
    'SampleMultinomial': 'Sampling', 'WarpPRNG': 'Sampling',
    'BatchSeqIdLookup': 'Sampling', 'BatchPerSeqArgmax': 'Sampling',
    'BatchPerSeqStopCheck': 'Sampling', 'SeqIdLookup': 'Sampling',
    # Misc
    'Comment': 'Misc', 'DebugBreakpoint': 'Misc', 'DebugMarker': 'Misc',
    'DebugProbe': 'Misc', 'DebugBreakIf': 'Misc', 'DeclareVReg': 'Misc',
    'ReleaseVReg': 'Misc', 'HotpatchSlot': 'Misc', 'LoadCallbackEntry': 'Misc',
    'NativeCall': 'Misc', 'Lz4Decode': 'Misc', 'BitPackRleDecode': 'Misc',
}

CATEGORIES = ['Memory', 'Arith', 'Control', 'Tile', 'Quant', 'GpuComm', 'Sampling', 'Misc']


def cat_snake(cat):
    """snake_case a category name: GpuComm -> gpu_comm, Misc -> misc."""
    s = re.sub(r'(?<!^)([A-Z])', r'_\1', cat).lower()
    return s

# nccl-gated variants (mirror vm_instr_category.rs)
NCCL_VARIANTS = {
    'AllReduceChunk', 'CommBarrier', 'NvlinkAsyncCopy', 'RemotePageLookup',
    'P2pPageFetch', 'RdmaPageFetch', 'RdmaPageFetchCompressed', 'RemotePageAttn',
    'PageMigrationLock', 'PageMigrationUnlock', 'PageLocationUpdate',
}


def parse_arms(src, fn_name='lower_instr'):
    """Parse the giant match in lower_instr. Returns list of arms.

    Each arm: {
        'cfg': '#[cfg(feature = "nccl")]' or '',
        'pattern': 'VmInstr::Variant { fields }' (full pattern text),
        'variant': 'Variant' (name only),
        'body': '... arm body text ...' (the block content inside {}),
        'body_full': '{ ... }' (full block including braces, or '=>' expression),
    }
    """
    # Find the match block start
    fn_idx = src.find('pub fn lower_instr(')
    if fn_idx < 0:
        fn_idx = src.find('fn lower_instr(')
    m_idx = src.find('match instr {', fn_idx)
    # The opening brace of match is at the { after 'match instr'
    open_brace = src.find('{', m_idx)

    arms = []
    i = open_brace + 1
    n = len(src)
    # We parse arms: each arm is [cfg]? <pattern> => <body>
    # pattern ends at ' =>' (top-level, brace depth 0 relative to match)
    # body is either a block { ... } or an expression up to top-level ','
    depth = 0
    while i < n:
        # skip whitespace/comments
        # Look for arm start: optional #[cfg(...)] attribute
        # then pattern until '=>'
        # then body until top-level ',' (depth 0) or closing '}' of match
        # Find next '=>' at the match's top level
        # First, gather the pattern region: from i until '=>'
        # But need to handle attributes
        arm_start = i
        # skip whitespace
        while i < n and src[i] in ' \t\n\r':
            i += 1
        if i >= n:
            break
        # closing brace of match?
        if src[i] == '}':
            break
        # parse pattern (may include leading comments + #[cfg] attributes)
        pat_start = i
        cfg = ''
        # strip leading // comment lines and capture #[cfg] attributes
        while i < n:
            # skip whitespace
            while i < n and src[i] in ' \t\n\r':
                i += 1
            if i >= n:
                break
            if src[i] == '/' and i + 1 < n and src[i+1] == '/':
                # line comment — skip to end of line
                while i < n and src[i] != '\n':
                    i += 1
                continue
            if src[i] == '#':
                # attribute #[cfg(...)] — capture
                attr_start = i
                depth_b = 0
                j = i
                while j < n:
                    if src[j] == '[':
                        depth_b += 1
                    elif src[j] == ']':
                        depth_b -= 1
                        if depth_b == 0:
                            j += 1
                            break
                    j += 1
                cfg_line = src[attr_start:j].strip()
                # only keep cfg attributes; ignore others (none expected here)
                if cfg_line.startswith('#[cfg'):
                    cfg = cfg_line
                i = j
                continue
            break
        # now pattern (VmInstr::...) until ' =>'
        pat_end = i
        # find '=>' at depth 0 (no nested braces in pattern generally, but patterns can have {..})
        d = 0
        while pat_end < n:
            c = src[pat_end]
            if c == '{':
                d += 1
            elif c == '}':
                d -= 1
            elif d == 0 and c == '=' and pat_end + 1 < n and src[pat_end+1] == '>':
                break
            pat_end += 1
        pattern = src[i:pat_end].strip()
        # advance past '=>'
        i = pat_end + 2
        # skip whitespace
        while i < n and src[i] in ' \t\n\r':
            i += 1
        # parse body
        if i < n and src[i] == '{':
            # block body
            bd = 1
            body_start = i + 1
            j = i + 1
            while j < n and bd > 0:
                c = src[j]
                if c == '{':
                    bd += 1
                elif c == '}':
                    bd -= 1
                    if bd == 0:
                        break
                elif c == '"':
                    # skip string literal
                    j += 1
                    while j < n and src[j] != '"':
                        if src[j] == '\\':
                            j += 1
                        j += 1
                elif c == "'":
                    # char literal or lifetime; skip conservatively
                    # only skip if it's a char literal 'x'
                    # lifetime: 'a — followed by ident
                    # heuristic: if next is not \\ and the char after closing ' is not ident
                    pass
                j += 1
            body = src[body_start:j]
            i = j + 1  # past closing }
        else:
            # expression body: until top-level ','
            j = i
            d = 0
            while j < n:
                c = src[j]
                if c == '{':
                    d += 1
                elif c == '}':
                    if d == 0:
                        # end of match
                        break
                    d -= 1
                elif c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                elif d == 0 and c == ',':
                    break
                elif c == '"':
                    j += 1
                    while j < n and src[j] != '"':
                        if src[j] == '\\':
                            j += 1
                        j += 1
                j += 1
            body = src[i:j].strip()
            i = j
        # extract variant name
        vm = re.match(r'VmInstr::(\w+)', pattern)
        variant = vm.group(1) if vm else None
        arms.append({
            'cfg': cfg,
            'pattern': pattern,
            'variant': variant,
            'body': body,
        })
        # skip trailing comma
        while i < n and src[i] in ' \t\n\r,':
            i += 1
    return arms, open_brace


def variant_to_fn(variant, isa):
    """lower_<VariantSnake>_<isa>"""
    # snake_case the variant name
    s = re.sub(r'(?<!^)([A-Z])', r'_\1', variant).lower()
    # fix: e.g. VmInstr::VecLoad -> vec_load
    return f'lower_{s}_{isa}'


def generate(isa, src_path):
    with open(src_path) as f:
        src = f.read()
    arms, match_open = parse_arms(src)

    # Validate: every arm's variant must be in CATEGORY_MAP
    missing = set()
    for a in arms:
        v = a['variant']
        if v not in CATEGORY_MAP:
            missing.add(v)
    if missing:
        print(f'ERROR: variants not in CATEGORY_MAP: {missing}', file=sys.stderr)
        sys.exit(1)

    # Group arms by category, preserving source order
    by_cat = {c: [] for c in CATEGORIES}
    for a in arms:
        by_cat[CATEGORY_MAP[a['variant']]].append(a)

    isa_lower = isa  # 'aarch64' or 'gpu'
    struct_name = {'aarch64': 'AArch64', 'gpu': 'Gpu'}[isa]

    # ── L0 dispatch (lower_instr_inner) ──
    # Note: x86 keeps ConditionalSkip preamble; aarch64/gpu may not have skip_stack.
    # We generate a clean L0 that delegates to category methods.
    # The original lower_instr wrapper (with error logging) is kept; we replace the match body.

    # ── L1 + L2 ──
    out = []
    out.append('impl %sLower {' % struct_name)
    out.append('')
    out.append('    // ════════════════════════════════════════════════════════════════════')
    out.append('    // ARCH-LOWER-DISPATCH-LAYERING (BCE-20260630-LOWER-INSTR-GOD-MATCH):')
    out.append('    // L1 变体路由 + L2 叶子 emit')
    out.append('    // lower_instr (L0) 按 VmInstr::category() 路由到 lower_<cat>_%s (L1),' % isa_lower)
    out.append('    // L1 内 match 变体路由到 lower_<variant>_%s (L2 叶子 fn)。' % isa_lower)
    out.append('    // 每个 L2 叶子 fn = 单变体作用域, 行为与原 god-match arm 完全等价 (行为保持)。')
    out.append('    // ════════════════════════════════════════════════════════════════════')
    out.append('')

    for cat in CATEGORIES:
        cat_arms = by_cat[cat]
        if not cat_arms:
            continue
        out.append('    /// L1 变体路由: %s 类 VmInstr → L2 叶子 emit fn。' % cat)
        out.append('    /// ISA 不支持的变体 (类别错配) 返回 Err (NO_SILENT_FALLBACK, 非 catch-all NOP)。')
        out.append('    fn lower_%s_%s(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {' % (cat_snake(cat), isa_lower))
        out.append('        match instr {')
        for a in cat_arms:
            v = a['variant']
            fn = variant_to_fn(v, isa_lower)
            cfg_prefix = (a['cfg'] + '\n            ') if a['cfg'] else ''
            out.append('            %s%s => self.%s(instr, alloc),' % (cfg_prefix, a['pattern'], fn))
        out.append('            _ => Err(CompilerError::CodegenViolation(')
        out.append('                format!("lower_%s_%s: variant {:?} not in %s category", instr))),' % (cat_snake(cat), isa_lower, cat))
        out.append('        }')
        out.append('    }')
        out.append('')

    # L2 leaf fns — group arms sharing the same fn name (multi-arm variant like GprBinOp
    # has 6 arms → 1 L2 fn with 6 match arms; multi-pattern arm like `A|B|C` → 1 arm).
    # Group key = fn name (derived from first variant). Preserves source order within group.
    out.append('    // ── L2 叶子 emit 函数 (单变体作用域, 行为与原 arm 等价) ──')
    out.append('')
    # Build ordered, de-duplicated fn list (first occurrence wins order).
    seen_fn = {}  # fn -> list of arms
    fn_order = []
    for cat in CATEGORIES:
        cat_arms = by_cat[cat]
        if not cat_arms:
            continue
        for a in cat_arms:
            v = a['variant']
            fn = variant_to_fn(v, isa_lower)
            if fn not in seen_fn:
                seen_fn[fn] = {'arms': [], 'cat': cat, 'first_variant': v}
                fn_order.append(fn)
            seen_fn[fn]['arms'].append(a)

    # Emit one L2 fn per group.
    last_cat = None
    for fn in fn_order:
        g = seen_fn[fn]
        cat = g['cat']
        v = g['first_variant']
        if cat != last_cat:
            out.append('    // ── %s ──' % cat)
            last_cat = cat
        # cfg attr on the fn: only if ALL arms share the same cfg (rare). Otherwise no attr
        # (per-arm cfg is applied inside match arms).
        cfgs = set(a['cfg'] for a in g['arms'])
        fn_cfg = ''
        if len(cfgs) == 1 and list(cfgs)[0]:
            fn_cfg = '    %s\n' % list(cfgs)[0]
        out.append('%s    fn %s(&mut self, instr: &VmInstr, alloc: &RegAllocation) -> Result<(), CompilerError> {' % (fn_cfg, fn))
        out.append('        match instr {')
        for a in g['arms']:
            cfg_prefix = ('            %s\n            ' % a['cfg']) if a['cfg'] else '            '
            out.append('%s%s => {' % (cfg_prefix, a['pattern']))
            out.append(a['body'].rstrip())
            out.append('            }')
        out.append('            _ => Err(CompilerError::CodegenViolation(')
        out.append('                format!("%s: expected VmInstr::%s, got {:?}", instr))),' % (fn, v))
        out.append('        }')
        out.append('    }')
        out.append('')

    out.append('}')

    return '\n'.join(out) + '\n', arms, by_cat


if __name__ == '__main__':
    isa = sys.argv[1]
    src_path = sys.argv[2]
    dispatch_text, arms, by_cat = generate(isa, src_path)
    print('=== ARMS PARSED: %d ===' % len(arms), file=sys.stderr)
    for cat in CATEGORIES:
        print('  %-9s: %d' % (cat, len(by_cat[cat])), file=sys.stderr)
    sys.stdout.write(dispatch_text)
