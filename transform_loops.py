#!/usr/bin/env python3
"""Transform remaining GEMM functions from m→n→chunk to chunk→m→n loop order."""

import re

FILE = "/home/putao/code/rust/gllm-kernels/src/macros/operator_templates.rs"

with open(FILE, "r") as f:
    content = f.read()

# Helper: generate the AVX2-style microkernel inner loop (6×16, 2 vecs, 8 lanes)
def avx2_microkernel_body(isa, tile_m, lanes, tile_n_vecs, use_prefetch=True):
    """Generate the FMA microkernel body for AVX2/AVX512 style."""
    TILE_N = tile_n_vecs * lanes
    rows = list(range(tile_m))

    # fma_row macro
    lines = []
    lines.append(f"                            macro_rules! fma_row {{")
    lines.append(f"                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {{")
    lines.append(f"                                    let va = $crate::simd_primitive!({isa}, $elem, splat, *$a_base.add($row));")
    lines.append(f"                                    $c0 = $crate::simd_primitive!({isa}, $elem, fma, va, $vb0, $c0);")
    lines.append(f"                                    $c1 = $crate::simd_primitive!({isa}, $elem, fma, va, $vb1, $c1);")
    lines.append(f"                                }};")
    lines.append(f"                            }}")

    # fma_all macro
    lines.append(f"                            macro_rules! fma_all {{")
    lines.append(f"                                ($a_base:expr, $vb0:ident, $vb1:ident) => {{")
    for r in rows:
        k_off = f"k_size * {r}" if r > 1 else ("k_size" if r == 1 else "0")
        lines.append(f"                                    fma_row!($a_base, $vb0, $vb1, {k_off}, c_{r}_0, c_{r}_1);")
    lines.append(f"                                }};")
    lines.append(f"                            }}")

    return "\n".join(lines)


def make_chunk_mn_loop(isa, tile_m, lanes, tile_n_vecs, is_bias, is_prepacked):
    """Generate the chunk→m→n main loop replacement."""
    TILE_N = tile_n_vecs * lanes
    rows = list(range(tile_m))
    macro_name = "fma_all" if isa == "avx2" else "fma_all_rows"

    lines = []

    # Init C
    if is_bias:
        lines.append(f"            // Init C with bias (each row gets the same bias vector)")
        lines.append(f"            let c_ptr = c.as_mut_ptr();")
        lines.append(f"            for m in 0..m_size {{")
        lines.append(f"                unsafe {{ std::ptr::copy_nonoverlapping(bias.as_ptr(), c_ptr.add(m * n_size), n_size); }}")
        lines.append(f"            }}")
    else:
        lines.append(f"            // Zero C matrix — accumulation across chunks")
        lines.append(f"            for i in 0..c.len().min(m_size * n_size) {{ c[i] = <$elem as Element>::ZERO; }}")
        lines.append(f"            let c_ptr = c.as_mut_ptr();")

    lines.append(f"")
    lines.append(f"            // Main loop: chunk → m → n (B stays hot in L1/L2)")
    lines.append(f"            let mut k_start = 0usize;")
    lines.append(f"            let mut chunk = 0usize;")
    lines.append(f"            while k_start < k_size {{")
    lines.append(f"                let kc = KC.min(k_size - k_start);")
    lines.append(f"")
    lines.append(f"                let mut m = 0;")
    lines.append(f"                while m + TILE_M <= m_size {{")
    lines.append(f"                    let mut n = 0;")
    lines.append(f"                    let mut strip_idx = 0;")
    lines.append(f"                    while n + TILE_N <= n_size {{")
    lines.append(f"                        unsafe {{")

    # Load C tile from memory
    lines.append(f"                            // Load C tile from memory (accumulate across chunks)")
    lines.append(f"                            macro_rules! load_row {{")
    lines.append(f"                                ($row:expr) => {{(")
    lines.append(f"                                    $crate::simd_primitive!({isa}, $elem, loadu, c_ptr.add((m + $row) * n_size + n)),")
    lines.append(f"                                    $crate::simd_primitive!({isa}, $elem, loadu, c_ptr.add((m + $row) * n_size + n + LANES)),")
    lines.append(f"                                )}};")
    lines.append(f"                            }}")
    for r in rows:
        lines.append(f"                            let (mut c_{r}_0, mut c_{r}_1) = load_row!({r});")
    lines.append(f"")

    # FMA macros
    lines.append(f"                            macro_rules! fma_row {{")
    lines.append(f"                                ($a_base:expr, $vb0:ident, $vb1:ident, $row:expr, $c0:ident, $c1:ident) => {{")
    lines.append(f"                                    let va = $crate::simd_primitive!({isa}, $elem, splat, *$a_base.add($row));")
    lines.append(f"                                    $c0 = $crate::simd_primitive!({isa}, $elem, fma, va, $vb0, $c0);")
    lines.append(f"                                    $c1 = $crate::simd_primitive!({isa}, $elem, fma, va, $vb1, $c1);")
    lines.append(f"                                }};")
    lines.append(f"                            }}")
    lines.append(f"                            macro_rules! {macro_name} {{")
    lines.append(f"                                ($a_base:expr, $vb0:ident, $vb1:ident) => {{")
    for r in rows:
        k_off = f"k_size * {r}" if r > 1 else ("k_size" if r == 1 else "0")
        lines.append(f"                                    fma_row!($a_base, $vb0, $vb1, {k_off}, c_{r}_0, c_{r}_1);")
    lines.append(f"                                }};")
    lines.append(f"                            }}")
    lines.append(f"")

    # A and B pointers
    lines.append(f"                            let mut a_col = a.as_ptr().add(m * k_size + k_start);")
    if is_prepacked:
        lines.append(f"                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);")
    else:
        lines.append(f"                            let mut b_ptr = packed_b.as_ptr().add(chunk * chunk_size + strip_idx * KC * TILE_N);")
    lines.append(f"")

    # K-loop unrolled 8x
    lines.append(f"                            let mut _k = 0usize;")
    lines.append(f"                            let k_unroll_end = kc & !7;")
    lines.append(f"                            while _k < k_unroll_end {{")
    lines.append(f"                                $crate::simd_primitive!({isa}, $elem, prefetch, b_ptr.add(TILE_N * 16) as *const i8, 0);")
    lines.append(f"")
    for u in range(8):
        lines.append(f"                                let vb{u}_0 = $crate::simd_primitive!({isa}, $elem, loadu, b_ptr{'.add(TILE_N * ' + str(u) + ')' if u > 0 else ''});")
        lines.append(f"                                let vb{u}_1 = $crate::simd_primitive!({isa}, $elem, loadu, b_ptr.add({'TILE_N * ' + str(u) + ' + ' if u > 0 else ''}LANES));")
        lines.append(f"                                {macro_name}!(a_col, vb{u}_0, vb{u}_1); a_col = a_col.add(1);")
        if u < 7:
            lines.append(f"")
    lines.append(f"")
    lines.append(f"                                b_ptr = b_ptr.add(TILE_N * 8);")
    lines.append(f"                                _k += 8;")
    lines.append(f"                            }}")
    lines.append(f"                            while _k < kc {{")
    lines.append(f"                                let vb_0 = $crate::simd_primitive!({isa}, $elem, loadu, b_ptr);")
    lines.append(f"                                let vb_1 = $crate::simd_primitive!({isa}, $elem, loadu, b_ptr.add(LANES));")
    lines.append(f"                                {macro_name}!(a_col, vb_0, vb_1); a_col = a_col.add(1);")
    lines.append(f"                                b_ptr = b_ptr.add(TILE_N);")
    lines.append(f"                                _k += 1;")
    lines.append(f"                            }}")
    lines.append(f"")

    # Store C tile back
    lines.append(f"                            macro_rules! store_row {{")
    lines.append(f"                                ($row:expr, $c0:expr, $c1:expr) => {{")
    lines.append(f"                                    $crate::simd_primitive!({isa}, $elem, storeu, c_ptr.add((m + $row) * n_size + n), $c0);")
    lines.append(f"                                    $crate::simd_primitive!({isa}, $elem, storeu, c_ptr.add((m + $row) * n_size + n + LANES), $c1);")
    lines.append(f"                                }};")
    lines.append(f"                            }}")

    # Store rows
    store_pairs = []
    for r in rows:
        store_pairs.append(f"store_row!({r}, c_{r}_0, c_{r}_1)")
    # Group 2 per line
    for i in range(0, len(store_pairs), 2):
        chunk_str = "; ".join(store_pairs[i:i+2]) + ";"
        lines.append(f"                            {chunk_str}")

    lines.append(f"                        }}")
    lines.append(f"                        n += TILE_N;")
    lines.append(f"                        strip_idx += 1;")
    lines.append(f"                    }}")
    lines.append(f"                    m += TILE_M;")
    lines.append(f"                }}")
    lines.append(f"")
    lines.append(f"                k_start += KC;")
    lines.append(f"                chunk += 1;")
    lines.append(f"            }}")
    lines.append(f"")

    # Remainder handling
    bias_init = "bias[n]" if is_bias else "<$elem as Element>::ZERO"
    if is_prepacked:
        b_access = "packed_b[chunk_idx * chunk_size + (n / TILE_N) * KC * TILE_N + k_in_chunk * TILE_N + (n % TILE_N)]"
        k_setup = "let chunk_idx = k / KC; let k_in_chunk = k % KC;"
    else:
        b_access = "b[k * n_size + n]"
        k_setup = ""

    lines.append(f"            // Remainder N (scalar) — columns not covered by TILE_N")
    lines.append(f"            let n_main = (n_size / TILE_N) * TILE_N;")
    lines.append(f"            for m in 0..m_size {{")
    lines.append(f"                for n in n_main..n_size {{")
    lines.append(f"                    let mut sum = {bias_init};")
    lines.append(f"                    for k in 0..k_size {{")
    if k_setup:
        lines.append(f"                        {k_setup}")
    lines.append(f"                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], {b_access});")
    lines.append(f"                    }}")
    lines.append(f"                    c[m * n_size + n] = sum;")
    lines.append(f"                }}")
    lines.append(f"            }}")
    lines.append(f"            // Remainder M (scalar) — rows not covered by TILE_M")
    lines.append(f"            let m_main = (m_size / TILE_M) * TILE_M;")
    lines.append(f"            for m in m_main..m_size {{")
    lines.append(f"                for n in 0..n_main {{")
    lines.append(f"                    let mut sum = {bias_init};")
    lines.append(f"                    for k in 0..k_size {{")
    if k_setup:
        lines.append(f"                        {k_setup}")
    lines.append(f"                        sum = <$elem as Element>::mul_add(sum, a[m * k_size + k], {b_access});")
    lines.append(f"                    }}")
    lines.append(f"                    c[m * n_size + n] = sum;")
    lines.append(f"                }}")
    lines.append(f"            }}")

    return "\n".join(lines)


def replace_function_body(content, start_marker, end_marker, new_body):
    """Replace the main loop section of a function between markers."""
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print(f"WARNING: Could not find start marker: {start_marker[:80]}")
        return content
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print(f"WARNING: Could not find end marker: {end_marker[:80]}")
        return content
    return content[:start_idx] + new_body + "\n" + content[end_idx:]


# ============================================================
# 1. AVX-512 matmul (non-prepacked)
# ============================================================
old_start = "            // Main loop: m → n → chunk (C stays in registers)\n"
old_end = "            WS.with(|c| c.set(packed_b));\n        }\n\n        #[inline(always)]\n        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem]"

new_body = make_chunk_mn_loop("avx512", 14, 16, 2, is_bias=False, is_prepacked=False)
new_body += "\n            WS.with(|c| c.set(packed_b));\n        }\n\n        #[inline(always)]\n        pub fn matmul(a: &[$elem], b: &[$elem], c: &mut [$elem]"

content = replace_function_body(content, old_start, old_end, new_body)

# ============================================================
# 2. AVX-512 matmul_bias (non-prepacked)
# ============================================================
old_start = "            // Main loop: m → n → chunk (C stays in registers, bias fused at init)\n"
old_end = "            WS.with(|c| c.set(packed_b));\n        }\n\n        #[inline(always)]\n        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem]"

new_body = make_chunk_mn_loop("avx512", 14, 16, 2, is_bias=True, is_prepacked=False)
new_body += "\n            WS.with(|c| c.set(packed_b));\n        }\n\n        #[inline(always)]\n        pub fn matmul_bias(a: &[$elem], b: &[$elem], bias: &[$elem]"

content = replace_function_body(content, old_start, old_end, new_body)

# ============================================================
# 3. AVX-512 matmul_prepacked
# ============================================================
# Find the main loop in matmul_prepacked_avx512
old_start_marker = "        /// Loop order: m → n → chunk(KC) — C stays in registers across KC chunks\n        #[target_feature(enable = \"avx512f\")]\n        unsafe fn matmul_prepacked_avx512"
idx = content.find(old_start_marker)
if idx != -1:
    # Find the main loop start (after chunk_size declaration)
    loop_search_start = content.find("            let mut m = 0;\n            while m + TILE_M <= m_size {", idx)
    loop_end = content.find("        }\n\n        #[inline(always)]\n        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem]", idx)
    if loop_search_start != -1 and loop_end != -1:
        new_body = make_chunk_mn_loop("avx512", 14, 16, 2, is_bias=False, is_prepacked=True)
        new_body += "\n        }\n\n        #[inline(always)]\n        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem]"
        content = content[:loop_search_start] + new_body + content[loop_end + len("        }\n\n        #[inline(always)]\n        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem]"):]
    else:
        print("WARNING: Could not find AVX-512 prepacked loop boundaries")
else:
    print("WARNING: Could not find AVX-512 matmul_prepacked_avx512")

# ============================================================
# 4. AVX-512 matmul_bias_prepacked
# ============================================================
old_start_marker = "        /// Matmul+bias using pre-packed B. Loop order: m → n → chunk(KC).\n        #[target_feature(enable = \"avx512f\")]\n        unsafe fn matmul_bias_prepacked_avx512"
idx = content.find(old_start_marker)
if idx != -1:
    loop_search_start = content.find("            let mut m = 0;\n            while m + TILE_M <= m_size {", idx)
    loop_end = content.find("        }\n\n        #[inline(always)]\n        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {\n            unsafe { matmul_bias_prepacked_avx512", idx)
    if loop_search_start != -1 and loop_end != -1:
        new_body = make_chunk_mn_loop("avx512", 14, 16, 2, is_bias=True, is_prepacked=True)
        end_str = "        }\n\n        #[inline(always)]\n        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {\n            unsafe { matmul_bias_prepacked_avx512"
        new_body += "\n" + end_str
        content = content[:loop_search_start] + new_body + content[loop_end + len(end_str):]
    else:
        print("WARNING: Could not find AVX-512 bias_prepacked loop boundaries")
else:
    print("WARNING: Could not find AVX-512 matmul_bias_prepacked_avx512")

# ============================================================
# 5. AVX2 matmul_prepacked
# ============================================================
old_start_marker = "        /// Matmul using pre-packed B (AVX2). Loop order: m → n → chunk(KC).\n        #[target_feature(enable = \"avx2\", enable = \"fma\")]\n        unsafe fn matmul_prepacked_avx2"
idx = content.find(old_start_marker)
if idx != -1:
    loop_search_start = content.find("            let mut m = 0;\n            while m + TILE_M <= m_size {", idx)
    loop_end = content.find("        }\n\n        #[inline(always)]\n        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {\n            unsafe { matmul_prepacked_avx2", idx)
    if loop_search_start != -1 and loop_end != -1:
        new_body = make_chunk_mn_loop("avx2", 6, 8, 2, is_bias=False, is_prepacked=True)
        end_str = "        }\n\n        #[inline(always)]\n        pub fn matmul_prepacked(a: &[$elem], packed_b: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {\n            unsafe { matmul_prepacked_avx2"
        new_body += "\n" + end_str
        content = content[:loop_search_start] + new_body + content[loop_end + len(end_str):]
    else:
        print("WARNING: Could not find AVX2 prepacked loop boundaries")
else:
    print("WARNING: Could not find AVX2 matmul_prepacked_avx2")

# ============================================================
# 6. AVX2 matmul_bias_prepacked
# ============================================================
old_start_marker = "        /// Matmul+bias using pre-packed B (AVX2). Loop order: m → n → chunk(KC).\n        #[target_feature(enable = \"avx2\", enable = \"fma\")]\n        unsafe fn matmul_bias_prepacked_avx2"
idx = content.find(old_start_marker)
if idx != -1:
    loop_search_start = content.find("            let mut m = 0;\n            while m + TILE_M <= m_size {", idx)
    loop_end = content.find("        }\n\n        #[inline(always)]\n        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {\n            unsafe { matmul_bias_prepacked_avx2", idx)
    if loop_search_start != -1 and loop_end != -1:
        new_body = make_chunk_mn_loop("avx2", 6, 8, 2, is_bias=True, is_prepacked=True)
        end_str = "        }\n\n        #[inline(always)]\n        pub fn matmul_bias_prepacked(a: &[$elem], packed_b: &[$elem], bias: &[$elem], c: &mut [$elem], m_size: usize, n_size: usize, k_size: usize) {\n            unsafe { matmul_bias_prepacked_avx2"
        new_body += "\n" + end_str
        content = content[:loop_search_start] + new_body + content[loop_end + len(end_str):]
    else:
        print("WARNING: Could not find AVX2 bias_prepacked loop boundaries")
else:
    print("WARNING: Could not find AVX2 matmul_bias_prepacked_avx2")

# Update comments
content = content.replace("Loop order: m → n → chunk(KC) — C stays in registers across KC chunks", "Loop order: chunk(KC) → m → n — B stays hot in cache")
content = content.replace("Loop order: m → n → chunk(KC).", "Loop order: chunk(KC) → m → n.")

with open(FILE, "w") as f:
    f.write(content)

print("Done! All transformations applied.")
