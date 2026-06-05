//! Scalar Gather and ColumnSlice — embedding lookup and column slicing.
//!
//! Phase 0 reference implementations for SymExec trace extraction.
//! Gather is an Injective op (index-based memory read).
//! ColumnSlice is an Injective op (row-stride-changing copy).

/// Gather: embedding lookup. `output[i, j] = table[indices[i], j]`.
///
/// - `indices`: `[seq_len]` f32 values representing integer token IDs
/// - `table`: `[table_rows, embed_dim]` row-major embedding table
/// - `output`: `[seq_len, embed_dim]` row-major output embeddings
///
/// Each index is truncated to usize (token IDs are integer values stored as f32).
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_gather(
    indices: *const f32,
    table: *const f32,
    output: *mut f32,
    seq_len: usize,
    embed_dim: usize,
    _table_rows: usize,
) {
    if seq_len == 0 || embed_dim == 0 || indices.is_null() || table.is_null() || output.is_null() {
        return;
    }
    unsafe {
        for i in 0..seq_len {
            let idx = *indices.add(i) as usize;
            let row_offset = idx * embed_dim;
            let out_offset = i * embed_dim;
            for j in 0..embed_dim {
                *output.add(out_offset + j) = *table.add(row_offset + j);
            }
        }
    }
}

/// ColumnSlice: row-major column slicing.
/// `output[s, j] = input[s, start + j]` for `s in [0, seq_len)`, `j in [0, slice_dim)`.
///
/// Input row stride = `input_inner`, output row stride = `slice_dim`.
/// Performs a real copy (row stride changes, cannot be zero-copy view).
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_column_slice(
    input: *const f32,
    output: *mut f32,
    seq_len: usize,
    input_inner: usize,
    start: usize,
    slice_dim: usize,
) {
    if seq_len == 0 || slice_dim == 0 || input.is_null() || output.is_null() {
        return;
    }
    unsafe {
        for s in 0..seq_len {
            let in_row = input.add(s * input_inner + start);
            let out_row = output.add(s * slice_dim);
            for j in 0..slice_dim {
                *out_row.add(j) = *in_row.add(j);
            }
        }
    }
}

/// QuantGather: quantized embedding lookup with on-the-fly dequantization.
///
/// Phase 0 scalar reference — defines the semantic of QuantGather for SymExec trace extraction.
/// The JIT codegen (emit_quant_gather_inline) replaces this with hardware-specific VmInstr.
///
/// Assumes GGUF-style interleaved block layout (scale header + packed nibbles per row).
/// Each row of the embed table is `row_blocks` consecutive quantized blocks.
/// Each block has `block_bytes` bytes: `header_bytes` of scale/metadata + `data_bytes` of packed values.
///
/// Parameters:
/// - `indices`:       `[seq_len]` i32 token IDs (cast to f32 for uniform ABI with scalar_gather).
/// - `table_quant`:   quantized embed table rows; each row = `(hidden_dim / block_size)` blocks.
/// - `output`:        `[seq_len, hidden_dim]` F32 output.
/// - `seq_len`:       number of tokens to look up.
/// - `hidden_dim`:    embedding dimension per row (must be divisible by `block_size`).
/// - `vocab_size`:    embed table row count (used for bounds checking only).
/// - `block_size`:    elements per quantization block (e.g., 32 for Q4_0).
/// - `block_bytes`:   total bytes per block (header + data; e.g., 18 for Q4_0: 2B scale + 16B nibbles).
/// - `header_bytes`:  bytes of scale/zero-point metadata at the start of each block.
///
/// This is a simplified scalar reference that decodes using linear dequant:
///   `value[i] = scale * (raw_nibble[i] - 8.0)`  (Q4_0 symmetric offset 8)
///
/// The actual JIT codegen uses `DecodeTraceBuilder` for format-specific decoding.
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn scalar_quant_gather(
    indices: *const f32,
    table_quant: *const u8,
    output: *mut f32,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
    block_size: usize,
    block_bytes: usize,
    header_bytes: usize,
) {
    if seq_len == 0 || hidden_dim == 0 || block_size == 0 || block_bytes == 0
        || indices.is_null() || table_quant.is_null() || output.is_null()
    {
        return;
    }
    let row_blocks = hidden_dim / block_size;
    let row_stride_bytes = row_blocks * block_bytes;
    unsafe {
        for i in 0..seq_len {
            let token_id = (*indices.add(i)) as usize;
            // Clamp to vocab_size to prevent OOB in scalar reference.
            let token_id = token_id.min(vocab_size.saturating_sub(1));
            let row_ptr = table_quant.add(token_id * row_stride_bytes);
            let out_ptr = output.add(i * hidden_dim);
            // Decode each block in the row.
            for blk in 0..row_blocks {
                let block_ptr = row_ptr.add(blk * block_bytes);
                // Read f16 scale from block header (bytes 0..2) — manual f16 → f32.
                let scale_bytes = [*block_ptr, *block_ptr.add(1)];
                let scale_bits = u16::from_le_bytes(scale_bytes);
                // Manual f16 → f32 conversion (IEEE 754 half-precision).
                let scale = {
                    let sign = ((scale_bits >> 15) as u32) << 31;
                    let exp = ((scale_bits >> 10) & 0x1F) as u32;
                    let mant = (scale_bits & 0x3FF) as u32;
                    let (exp32, mant32) = if exp == 0 {
                        (0u32, mant << 13)
                    } else if exp == 31 {
                        (0xFF, mant << 13)
                    } else {
                        (exp + 127 - 15, mant << 13)
                    };
                    f32::from_bits(sign | (exp32 << 23) | mant32)
                };
                // Packed data starts at offset `header_bytes`.
                let data_ptr = block_ptr.add(header_bytes);
                let elem_base = blk * block_size;
                // Decode nibbles: each byte encodes two 4-bit values (low nibble = even, high = odd).
                for j in 0..(block_size / 2) {
                    let byte = *data_ptr.add(j);
                    let lo = (byte & 0x0F) as f32 - 8.0;
                    let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                    *out_ptr.add(elem_base + j * 2)     = scale * lo;
                    *out_ptr.add(elem_base + j * 2 + 1) = scale * hi;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_gather_basic() {
        // Table: 3 rows x 4 cols
        let table = vec![
            1.0_f32, 2.0, 3.0, 4.0,  // row 0
            5.0, 6.0, 7.0, 8.0,      // row 1
            9.0, 10.0, 11.0, 12.0,   // row 2
        ];
        let indices = vec![0.0_f32, 2.0, 1.0]; // lookup rows 0, 2, 1
        let mut output = vec![0.0_f32; 12];

        scalar_gather(
            indices.as_ptr(),
            table.as_ptr(),
            output.as_mut_ptr(),
            3,   // seq_len
            4,   // embed_dim
            3,   // table_rows
        );

        let expected = vec![
            1.0, 2.0, 3.0, 4.0,    // row 0
            9.0, 10.0, 11.0, 12.0, // row 2
            5.0, 6.0, 7.0, 8.0,    // row 1
        ];
        for i in 0..12 {
            assert!(
                (output[i] - expected[i]).abs() < 1e-6,
                "gather[{i}]: got {}, expected {}",
                output[i], expected[i]
            );
        }
    }

    #[test]
    fn test_scalar_column_slice_basic() {
        // Input: 3 rows x 6 cols
        let input = vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0,   // row 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,     // row 1
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,  // row 2
        ];
        let mut output = vec![0.0_f32; 6]; // 3 rows x 2 cols

        scalar_column_slice(
            input.as_ptr(),
            output.as_mut_ptr(),
            3,   // seq_len
            6,   // input_inner
            2,   // start
            2,   // slice_dim
        );

        // Expected: columns [2, 3] from each row
        let expected = vec![
            3.0, 4.0,
            9.0, 10.0,
            15.0, 16.0,
        ];
        for i in 0..6 {
            assert!(
                (output[i] - expected[i]).abs() < 1e-6,
                "column_slice[{i}]: got {}, expected {}",
                output[i], expected[i]
            );
        }
    }
}
