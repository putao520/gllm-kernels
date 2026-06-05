//! Integration tests — CPU kernels, quantization, assembly GEMM.

// 代码组织 (include! 模式):
// - `tests_fragments/part1.inc.rs` — Block 1-18 (basic kernels through Q8K dot accuracy)
// - `tests_fragments/part2.inc.rs` — Block 19-29 (Q4K through assembly GEMM)

#[cfg(test)]
mod tests {
    use crate::cpu_kernels::CpuKernels;
    use crate::traits::Kernels;

    include!("tests_fragments/part1.inc.rs");
    include!("tests_fragments/part2.inc.rs");
}
