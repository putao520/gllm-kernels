//! Test that quantization IR can be activated in the graph optimization pipeline.

#[cfg(test)]
mod tests {
    use crate::compiler::{CompilerGraph, OpKind};
    use crate::compiler::quant_ir::QuantFormat;
    use crate::compiler::quant_convert;
    use crate::types::DType;

    #[test]
    fn test_apply_quantization_int8() {
        // Create a minimal graph with a GEMM op
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 1024], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[1024, 512], DType::F32);
        let output = graph.add_tensor_concrete("output", &[1, 512], DType::F32);
        let _op = graph.add_op(
            OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 512,
                k: 1024,
                dtype: DType::F32, trans_b: false },
            vec![input, weight],
            vec![output],
            "gemm",
        );

        // Apply Int8 quantization (implemented format)
        let result = quant_convert::apply_quantization(&mut graph, QuantFormat::Int8PerTensor);

        // Should succeed
        assert!(result.is_ok(), "Int8 quantization failed: {:?}", result.err());
    }

    #[test]
    fn test_apply_quantization_fp8() {
        // Create a minimal graph with a GEMM op
        let mut graph = CompilerGraph::new();
        let input = graph.add_tensor_concrete("input", &[1, 2048], DType::F32);
        let weight = graph.add_tensor_concrete("weight", &[2048, 1024], DType::F32);
        let output = graph.add_tensor_concrete("output", &[1, 1024], DType::F32);
        let _op = graph.add_op(
            OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 1024,
                k: 2048,
                dtype: DType::F32, trans_b: false },
            vec![input, weight],
            vec![output],
            "gemm",
        );

        // Apply FP8 E4M3 quantization (implemented format)
        let result = quant_convert::apply_quantization(&mut graph, QuantFormat::Fp8E4M3);

        // Should succeed
        assert!(result.is_ok(), "FP8 quantization failed: {:?}", result.err());
    }

    #[test]
    fn test_apply_quantization_multiple_gemms() {
        // Create a graph with multiple GEMM ops
        let mut graph = CompilerGraph::new();

        // First GEMM
        let input1 = graph.add_tensor_concrete("input1", &[1, 512], DType::F32);
        let weight1 = graph.add_tensor_concrete("weight1", &[512, 256], DType::F32);
        let output1 = graph.add_tensor_concrete("output1", &[1, 256], DType::F32);
        let _op1 = graph.add_op(
            OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 256,
                k: 512,
                dtype: DType::F32, trans_b: false },
            vec![input1, weight1],
            vec![output1],
            "gemm1",
        );

        // Second GEMM
        let input2 = graph.add_tensor_concrete("input2", &[1, 256], DType::F32);
        let weight2 = graph.add_tensor_concrete("weight2", &[256, 128], DType::F32);
        let output2 = graph.add_tensor_concrete("output2", &[1, 128], DType::F32);
        let _op2 = graph.add_op(
            OpKind::Gemm {
                m: crate::compiler::graph::SymDim::Concrete(1),
                n: 128,
                k: 256,
                dtype: DType::F32, trans_b: false },
            vec![input2, weight2],
            vec![output2],
            "gemm2",
        );

        // Apply quantization to all GEMMs
        let result = quant_convert::apply_quantization(&mut graph, QuantFormat::Int8PerTensor);

        // Should succeed
        assert!(result.is_ok(), "Multi-GEMM quantization failed: {:?}", result.err());
    }
}
