//! Page byte-stream codec JIT decoders (per SPEC 22-PAGE-COMPRESSION §3.3).
//!
//! This module provides helper functions that emit `VmInstr::Lz4Decode` and
//! `VmInstr::BitPackRleDecode` into a `VmProgram`.  The actual ISA lowering
//! (x86_64 / AArch64 / GPU) lives in the respective `*_lower.rs` files.
//!
//! REQ-COMP-003: LZ4 JIT 解压
//! REQ-COMP-004: BitPackRle JIT 解压

use super::instr::{VmInstr, VmProgram, VRegId};

/// Emit an LZ4 stream decode instruction into `prog`.
///
/// At runtime the generated code reads `compressed_size` bytes from `src_ptr`
/// and writes the decompressed page bytes to `dst_ptr`.
///
/// # Arguments
/// - `prog` — target VmProgram
/// - `src_ptr` — GPR VReg pointing to the compressed byte stream
/// - `dst_ptr` — GPR VReg pointing to the destination (page physical address)
/// - `compressed_size` — GPR VReg containing the byte count of the compressed stream
///   (read from `KvPageHeader.compressed_size`)
/// - `decompressed_size` — compile-time page size in bytes (e.g. `page_tokens * elem_bytes`)
pub fn emit_lz4_decode(
    prog: &mut VmProgram,
    src_ptr: VRegId,
    dst_ptr: VRegId,
    compressed_size: VRegId,
    decompressed_size: usize,
) {
    prog.emit(VmInstr::Lz4Decode {
        src_ptr,
        dst_ptr,
        compressed_size,
        decompressed_size,
    });
}

/// Emit a BitPackRle stream decode instruction into `prog`.
///
/// The nibble stream format is:
///   `[byte_i]` — each byte encodes one run:
///   - low nibble  = run_value
///   - high nibble = run_len  (==15 → escape: subsequent bytes are additional length until <255)
///
/// # Arguments
/// - `prog` — target VmProgram
/// - `src_ptr` — GPR VReg pointing to the compressed nibble stream
/// - `dst_ptr` — GPR VReg pointing to the output buffer
/// - `compressed_size` — GPR VReg containing the byte count of the compressed stream
/// - `nibble_bits` — bit-width per element: 4 for KIVI4, 2 for KIVI2
/// - `element_count` — number of output elements expected (page_size)
pub fn emit_bitpack_rle_decode(
    prog: &mut VmProgram,
    src_ptr: VRegId,
    dst_ptr: VRegId,
    compressed_size: VRegId,
    nibble_bits: u8,
    element_count: usize,
) {
    prog.emit(VmInstr::BitPackRleDecode {
        src_ptr,
        dst_ptr,
        compressed_size,
        nibble_bits,
        element_count,
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Reference encoder helpers (CPU only, used by unit tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Hand-roll a minimal LZ4 compressor for small test inputs.
///
/// This produces a valid LZ4 block stream that can be decompressed by the
/// JIT decoder. Encodes the **entire** input as a single literal-only last
/// sequence (no back-references), which is always valid in the LZ4 block
/// format because the last sequence is permitted to omit the match section.
///
/// Format: `[token][literal_len_ext...][literal_bytes]`
/// where token high nibble = min(lit_len, 15) and extension bytes encode the
/// remainder when lit_len >= 15.
pub fn lz4_compress_literals_only(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let lit_len = input.len();

    if lit_len <= 14 {
        // Fits in token high nibble directly
        out.push((lit_len as u8) << 4);
    } else {
        // Token high nibble = 15; remainder encoded as extension bytes
        out.push(0xF0u8);
        let mut remaining = lit_len - 15;
        while remaining >= 255 {
            out.push(0xFFu8);
            remaining -= 255;
        }
        out.push(remaining as u8);
    }
    // Copy all literal bytes; no match offset/length (last sequence)
    out.extend_from_slice(input);
    out
}

/// Hand-roll a minimal BitPackRle compressor.
///
/// Scans `input` and produces [low=run_value, high=run_len] byte stream.
/// Handles run_len extension (==15 escape) correctly.
pub fn bitpack_rle_compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    if input.is_empty() {
        return out;
    }
    let mut i = 0usize;
    while i < input.len() {
        let val = input[i] & 0x0F; // treat each byte as a nibble value
        let mut run_len = 1usize;
        while i + run_len < input.len() && (input[i + run_len] & 0x0F) == val && run_len < 65535 {
            run_len += 1;
        }

        // Encode: [val (low nibble), run_len (high nibble, with escape)]
        if run_len <= 14 {
            out.push(val | ((run_len as u8) << 4));
        } else {
            // initial nibble = 15 (escape)
            out.push(val | 0xF0u8);
            let mut rem = run_len - 15;
            // Write 255-bytes until < 255
            while rem >= 255 {
                out.push(0xFFu8);
                rem -= 255;
            }
            out.push(rem as u8);
        }

        i += run_len;
    }
    out
}

/// CPU reference decoder for LZ4 (mirrors JIT semantics).
///
/// Returns the decompressed bytes or an error string.
pub fn lz4_decode_reference(compressed: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut src = 0usize;

    while src < compressed.len() {
        let token = compressed[src] as usize;
        src += 1;

        // Literal length
        let mut lit_len = token >> 4;
        if lit_len == 15 {
            loop {
                if src >= compressed.len() {
                    return Err("truncated literal extension".into());
                }
                let ext = compressed[src] as usize;
                src += 1;
                lit_len += ext;
                if ext < 255 { break; }
            }
        }

        // Copy literals
        if src + lit_len > compressed.len() {
            return Err(format!("literal overrun: need {} bytes at pos {}", lit_len, src));
        }
        out.extend_from_slice(&compressed[src..src + lit_len]);
        src += lit_len;

        // End of last sequence (no match)
        if src >= compressed.len() {
            break;
        }

        // Match offset (LE u16)
        if src + 2 > compressed.len() {
            return Err("truncated match offset".into());
        }
        let moff = (compressed[src] as usize) | ((compressed[src + 1] as usize) << 8);
        src += 2;

        // Match length
        let mut match_len = (token & 0xF) + 4;
        if (token & 0xF) == 15 {
            loop {
                if src >= compressed.len() {
                    return Err("truncated match extension".into());
                }
                let ext = compressed[src] as usize;
                src += 1;
                match_len += ext;
                if ext < 255 { break; }
            }
        }

        // Copy match (may overlap)
        if moff == 0 || out.len() < moff {
            return Err(format!("invalid match offset {} at dst_pos {}", moff, out.len()));
        }
        let match_start = out.len() - moff;
        for k in 0..match_len {
            let byte = out[match_start + k];
            out.push(byte);
        }
    }
    Ok(out)
}

/// CPU reference decoder for BitPackRle (mirrors JIT semantics).
pub fn bitpack_rle_decode_reference(compressed: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut src = 0usize;
    while src < compressed.len() {
        let byte = compressed[src];
        src += 1;
        let val = byte & 0x0F;
        let mut run_len = (byte >> 4) as usize;
        if run_len == 15 {
            loop {
                if src >= compressed.len() { break; }
                let ext = compressed[src] as usize;
                src += 1;
                run_len += ext;
                if ext < 255 { break; }
            }
        }
        for _ in 0..run_len {
            out.push(val);
        }
    }
    out
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Unit Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::vm::instr::{VmProgram, VRegKind, SimdWidth};

    /// Verify that `emit_lz4_decode` appends a `Lz4Decode` instruction to the program.
    #[test]
    fn emit_lz4_decode_appends_instr() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_lz4_decode(&mut prog, src, dst, csz, 4096);

        // There should be 3 DeclareVReg + 1 Lz4Decode
        let lz4_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Lz4Decode { .. })).count();
        assert_eq!(lz4_count, 1, "expected exactly 1 Lz4Decode instruction");
    }

    /// Verify that `emit_bitpack_rle_decode` appends a `BitPackRleDecode` instruction.
    #[test]
    fn emit_bitpack_rle_decode_appends_instr() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_bitpack_rle_decode(&mut prog, src, dst, csz, 4, 256);

        let rle_count = prog.instrs.iter()
            .filter(|i| matches!(i, VmInstr::BitPackRleDecode { .. }))
            .count();
        assert_eq!(rle_count, 1, "expected exactly 1 BitPackRleDecode instruction");
    }

    /// LZ4 reference encoder + decoder round-trip with minimal data.
    #[test]
    fn lz4_roundtrip_basic() {
        // 256 bytes of structured data (some patterns to verify correctness)
        let original: Vec<u8> = (0u8..=255u8).collect();

        // Compress using literals-only encoder
        let compressed = lz4_compress_literals_only(&original);

        // Decompress using reference decoder
        let decompressed = lz4_decode_reference(&compressed)
            .expect("lz4 reference decode should succeed");

        assert_eq!(
            decompressed, original,
            "LZ4 round-trip failed: decompressed output does not match original"
        );
    }

    /// LZ4 round-trip with a larger page-sized buffer (4096 bytes).
    #[test]
    fn lz4_roundtrip_page_size() {
        // Create a realistic KV-cache-like page: mix of repeated patterns and unique data
        let mut original = Vec::with_capacity(4096);
        for i in 0..4096usize {
            original.push(((i * 7 + i / 256) & 0xFF) as u8);
        }

        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed)
            .expect("lz4 page-size round-trip should succeed");

        assert_eq!(decompressed, original, "LZ4 4096-byte round-trip failed");
    }

    /// BitPackRle reference encoder + decoder round-trip for KIVI4 nibble stream.
    #[test]
    fn bitpack_rle_kivi4_roundtrip() {
        // KIVI4 nibble stream: 256 nibble values (4-bit each, stored as bytes 0..=15)
        let original: Vec<u8> = (0u8..=15u8)
            .cycle()
            .take(256)
            .collect();

        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);

        assert_eq!(
            decompressed, original,
            "BitPackRle KIVI4 round-trip failed"
        );
    }

    /// BitPackRle round-trip with long runs (triggers extension encoding).
    #[test]
    fn bitpack_rle_long_run_roundtrip() {
        // 300 zeros — forces run_len=15 extension (15 + 255 + 30 = 300)
        let original = vec![0u8; 300];

        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);

        assert_eq!(decompressed, original, "BitPackRle long-run round-trip failed");
        // Compressed should be much smaller than original
        assert!(
            compressed.len() < original.len(),
            "BitPackRle compression should shrink repeated data: compressed={}, original={}",
            compressed.len(), original.len()
        );
    }

    /// Verify VmInstr field values are correctly set by emit helpers.
    #[test]
    fn lz4_decode_instr_fields() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_lz4_decode(&mut prog, src, dst, csz, 8192);

        let instr = prog.instrs.iter()
            .find(|i| matches!(i, VmInstr::Lz4Decode { .. }))
            .expect("Lz4Decode instruction not found");

        if let VmInstr::Lz4Decode { src_ptr, dst_ptr, compressed_size, decompressed_size } = instr {
            assert_eq!(*src_ptr, src);
            assert_eq!(*dst_ptr, dst);
            assert_eq!(*compressed_size, csz);
            assert_eq!(*decompressed_size, 8192);
        }
    }

    /// Verify VmInstr field values for BitPackRleDecode.
    #[test]
    fn bitpack_rle_decode_instr_fields() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_bitpack_rle_decode(&mut prog, src, dst, csz, 2, 512);

        let instr = prog.instrs.iter()
            .find(|i| matches!(i, VmInstr::BitPackRleDecode { .. }))
            .expect("BitPackRleDecode instruction not found");

        if let VmInstr::BitPackRleDecode {
            src_ptr, dst_ptr, compressed_size, nibble_bits, element_count
        } = instr {
            assert_eq!(*src_ptr, src);
            assert_eq!(*dst_ptr, dst);
            assert_eq!(*compressed_size, csz);
            assert_eq!(*nibble_bits, 2u8);
            assert_eq!(*element_count, 512);
        }
    }

    // ── Additional tests ──────────────────────────────────────────────

    /// LZ4 literals-only compressor: empty input produces a valid (minimal) stream.
    #[test]
    fn lz4_compress_empty_input() {
        let compressed = lz4_compress_literals_only(&[]);

        // Token byte with lit_len=0 → token = 0x00, no literals follow
        assert_eq!(compressed, vec![0x00u8], "empty input should produce [0x00] token");
    }

    /// LZ4 round-trip for input shorter than 15 bytes (fits in token nibble).
    #[test]
    fn lz4_roundtrip_short() {
        let original = vec![0xABu8, 0xCD, 0xEF, 0x12, 0x34];

        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed)
            .expect("lz4 short input round-trip should succeed");

        // Verify token encodes literal length directly (no extension bytes)
        assert_eq!(compressed[0] >> 4, original.len() as u8,
            "token high nibble should equal literal count for short input");
        assert_eq!(decompressed, original, "LZ4 short round-trip output mismatch");
    }

    /// LZ4 round-trip for input exactly 15 bytes (boundary: triggers extension).
    #[test]
    fn lz4_roundtrip_exactly_15() {
        let original: Vec<u8> = (0..15).collect();

        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed)
            .expect("lz4 15-byte round-trip should succeed");

        // At 15 bytes, token high nibble = 15 and extension = 0
        assert_eq!(compressed[0] & 0xF0, 0xF0, "token should indicate extension");
        assert_eq!(compressed[1], 0u8, "extension byte should be 0 for exactly 15 literals");
        assert_eq!(decompressed, original, "LZ4 15-byte round-trip output mismatch");
    }

    /// LZ4 round-trip for a larger input that needs multiple extension bytes (>= 270 literals).
    #[test]
    fn lz4_roundtrip_multi_extension() {
        // 300 bytes: 15 (token) + 255 (first ext) + 30 (second ext)
        let original: Vec<u8> = (0..300).map(|i| (i & 0xFF) as u8).collect();

        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed)
            .expect("lz4 multi-extension round-trip should succeed");

        // Verify extension encoding: 0xFF then remaining 30
        assert_eq!(compressed[1], 0xFF, "first extension byte should be 255");
        assert_eq!(compressed[2], 30u8, "second extension byte should be 30");
        assert_eq!(decompressed, original, "LZ4 multi-extension round-trip output mismatch");
    }

    /// LZ4 reference decoder rejects truncated literal extension bytes.
    #[test]
    fn lz4_decode_truncated_extension() {
        // Token says 15+extension but no extension byte follows
        let bad = vec![0xF0u8];

        let result = lz4_decode_reference(&bad);
        assert!(result.is_err(), "should reject truncated literal extension");
        assert!(
            result.unwrap_err().contains("truncated"),
            "error message should mention truncation"
        );
    }

    /// LZ4 reference decoder rejects literal overrun (not enough source bytes).
    #[test]
    fn lz4_decode_literal_overrun() {
        // Token says 3 literals but only 1 byte follows
        let bad = vec![0x30u8, 0xAA];

        let result = lz4_decode_reference(&bad);
        assert!(result.is_err(), "should reject literal overrun");
        assert!(
            result.unwrap_err().contains("overrun"),
            "error message should mention overrun"
        );
    }

    /// BitPackRle compressor: empty input produces empty output.
    #[test]
    fn bitpack_rle_compress_empty() {
        let compressed = bitpack_rle_compress(&[]);
        assert!(compressed.is_empty(), "empty input should produce empty output");
    }

    /// BitPackRle round-trip with all identical values (maximum compression).
    #[test]
    fn bitpack_rle_single_value_roundtrip() {
        let original = vec![7u8; 100];

        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);

        // 100 identical values: run_len > 14, needs extension (15 + 85)
        assert!(
            compressed.len() < 10,
            "single-value stream should compress heavily: got {} bytes",
            compressed.len()
        );
        assert_eq!(decompressed, original, "BitPackRle single-value round-trip mismatch");
    }

    /// BitPackRle round-trip with alternating values (worst case: no runs).
    #[test]
    fn bitpack_rle_alternating_values_roundtrip() {
        let original: Vec<u8> = (0..200).map(|i| (i % 2) as u8).collect();

        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);

        // Alternating 0,1,0,1... each run is length 1, so compressed is same size as input
        assert_eq!(
            compressed.len(),
            original.len(),
            "alternating values should have 1:1 compression ratio"
        );
        assert_eq!(decompressed, original, "BitPackRle alternating round-trip mismatch");
    }

    /// BitPackRle decode reference handles a single byte encoding run_len < 15.
    #[test]
    fn bitpack_rle_decode_single_byte() {
        // One byte: value=3 (low nibble), run_len=5 (high nibble)
        let compressed = vec![0x53u8];
        let decompressed = bitpack_rle_decode_reference(&compressed);

        assert_eq!(decompressed, vec![3u8; 5], "single encoded byte should expand to 5 copies of value 3");
    }

    /// Multiple emit calls accumulate instructions correctly in VmProgram.
    #[test]
    fn emit_multiple_decodes_accumulates() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_lz4_decode(&mut prog, src, dst, csz, 4096);
        emit_bitpack_rle_decode(&mut prog, src, dst, csz, 4, 128);
        emit_lz4_decode(&mut prog, src, dst, csz, 8192);

        let lz4_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::Lz4Decode { .. })).count();
        let rle_count = prog.instrs.iter().filter(|i| matches!(i, VmInstr::BitPackRleDecode { .. })).count();

        assert_eq!(lz4_count, 2, "expected 2 Lz4Decode instructions");
        assert_eq!(rle_count, 1, "expected 1 BitPackRleDecode instruction");
    }

    // ── Additional tests ──────────────────────────────────────────────

    /// LZ4 round-trip with all-zero bytes (common KV cache pattern).
    #[test]
    fn lz4_roundtrip_all_zeros() {
        let original = vec![0u8; 1024];
        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    /// LZ4 round-trip with single byte input.
    #[test]
    fn lz4_roundtrip_single_byte() {
        let original = vec![0xFFu8];
        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed).unwrap();
        assert_eq!(decompressed, original);
        assert_eq!(compressed[0] >> 4, 1);
    }

    /// LZ4 round-trip for exactly 270 bytes (triggers two extension bytes: 15 + 255 + 0).
    #[test]
    fn lz4_roundtrip_exactly_270() {
        let original: Vec<u8> = (0..270).map(|i| (i & 0xFF) as u8).collect();
        let compressed = lz4_compress_literals_only(&original);
        let decompressed = lz4_decode_reference(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    /// LZ4 reference decoder rejects truncated match offset.
    #[test]
    fn lz4_decode_truncated_match_offset() {
        let bad = vec![0x0Fu8, 0xAA];
        let result = lz4_decode_reference(&bad);
        assert!(result.is_err(), "should reject truncated match offset");
    }

    /// LZ4 reference decoder rejects zero match offset.
    #[test]
    fn lz4_decode_zero_match_offset() {
        let bad = vec![0x05u8, 0x00, 0x00];
        let result = lz4_decode_reference(&bad);
        assert!(result.is_err(), "should reject zero match offset");
    }

    /// BitPackRle compressor handles values with high nibble set (only low nibble used).
    #[test]
    fn bitpack_rle_low_nibble_mask() {
        let original: Vec<u8> = vec![0x1A, 0x1A, 0x1A];
        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);
        assert!(decompressed.iter().all(|&v| v == 0x0A), "values should be masked to low nibble");
    }

    /// BitPackRle round-trip with KIVI2 nibble width (2-bit values).
    #[test]
    fn bitpack_rle_kivi2_values() {
        let original: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3].repeat(16);
        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);
        assert_eq!(decompressed, original);
    }

    /// BitPackRle extension encoding for run_len > 255.
    #[test]
    fn bitpack_rle_very_long_run() {
        let original = vec![5u8; 600];
        let compressed = bitpack_rle_compress(&original);
        let decompressed = bitpack_rle_decode_reference(&compressed);
        assert_eq!(decompressed, original);
        assert!(compressed.len() < 10, "long run should compress heavily");
    }

    /// emit_lz4_decode with zero decompressed_size.
    #[test]
    fn emit_lz4_decode_zero_size() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_lz4_decode(&mut prog, src, dst, csz, 0);

        let instr = prog.instrs.iter()
            .find(|i| matches!(i, VmInstr::Lz4Decode { .. }))
            .expect("Lz4Decode should exist");
        if let VmInstr::Lz4Decode { decompressed_size, .. } = instr {
            assert_eq!(*decompressed_size, 0);
        }
    }

    /// emit_bitpack_rle_decode with nibble_bits=2.
    #[test]
    fn emit_bitpack_rle_decode_nibble_bits_2() {
        let mut prog = VmProgram::new();
        let src = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let dst = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);
        let csz = prog.alloc_vreg(VRegKind::Ptr, SimdWidth::Scalar);

        emit_bitpack_rle_decode(&mut prog, src, dst, csz, 2, 1024);

        let instr = prog.instrs.iter()
            .find(|i| matches!(i, VmInstr::BitPackRleDecode { .. }))
            .expect("BitPackRleDecode should exist");
        if let VmInstr::BitPackRleDecode { nibble_bits, element_count, .. } = instr {
            assert_eq!(*nibble_bits, 2);
            assert_eq!(*element_count, 1024);
        }
    }

    /// BitPackRle decode reference with empty compressed input produces empty output.
    #[test]
    fn bitpack_rle_decode_empty_input() {
        let decompressed = bitpack_rle_decode_reference(&[]);
        assert!(decompressed.is_empty());
    }
}
