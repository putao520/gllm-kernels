use gllm_kernels::quantization::{
    dequantize_blockwise_int1, dequantize_blockwise_int2, matmul_blockwise_int1,
    matmul_blockwise_int2, quantize_blockwise_int1, quantize_blockwise_int2, PackedU8,
};
use gllm_kernels::{DType, PackedBits};

#[test]
fn packed_u8_and_dtype_storage() {
    let values = vec![1i8, -1, 1, -1, -1, 1, -1, 1];
    let packed = PackedU8::from_i8(&values, PackedBits::Int1).unwrap();
    assert_eq!(packed.values_per_byte(), 8);

    let dtype = DType::PackedU8(PackedBits::Int1);
    assert_eq!(dtype.storage_bytes_for(values.len()), Some(1));
    assert_eq!(dtype.bits_per_value(), 1);
}

#[test]
fn blockwise_int2_matmul_matches_dequantized() {
    let rows = 3;
    let cols = 6;
    let input_rows = 2;
    let input: Vec<f32> = (0..input_rows * cols)
        .map(|i| (i as f32 - 3.0) * 0.25)
        .collect();
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.2) - 1.0).collect();

    let matrix = quantize_blockwise_int2::<16>(&weight, rows, cols).unwrap();
    let mut dequant = vec![0f32; rows * cols];
    dequantize_blockwise_int2(&matrix, &mut dequant).unwrap();

    let mut expected = vec![0f32; input_rows * rows];
    for i in 0..input_rows {
        for j in 0..rows {
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += input[i * cols + k] * dequant[j * cols + k];
            }
            expected[i * rows + j] = sum;
        }
    }

    let mut output = vec![0f32; input_rows * rows];
    matmul_blockwise_int2(&input, &matrix, &mut output, input_rows).unwrap();

    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-3);
    }
}

#[test]
fn blockwise_int1_matmul_matches_dequantized() {
    let rows = 2;
    let cols = 5;
    let input_rows = 2;
    let input: Vec<f32> = (0..input_rows * cols)
        .map(|i| (i as f32 - 2.0) * 0.3)
        .collect();
    let weight: Vec<f32> = (0..rows * cols)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let matrix = quantize_blockwise_int1::<16>(&weight, rows, cols).unwrap();
    let mut dequant = vec![0f32; rows * cols];
    dequantize_blockwise_int1(&matrix, &mut dequant).unwrap();

    let mut expected = vec![0f32; input_rows * rows];
    for i in 0..input_rows {
        for j in 0..rows {
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += input[i * cols + k] * dequant[j * cols + k];
            }
            expected[i * rows + j] = sum;
        }
    }

    let mut output = vec![0f32; input_rows * rows];
    matmul_blockwise_int1(&input, &matrix, &mut output, input_rows).unwrap();

    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-3);
    }
}
