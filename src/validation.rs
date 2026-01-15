//! Zero-cost validation utilities for GPU kernel parameters.
//!
//! This module provides compile-time and runtime validation for attention and embedding
//! kernel parameters. All validation functions are generic over the error type to allow
//! each backend to use its own error enum while sharing validation logic.
//!
//! # Design
//!
//! - All functions return `Result<T, String>` for flexible error conversion
//! - Each backend maps String errors to its own error type
//! - Overflow checks use `checked_mul` for safety
//! - const MAX values defined once, used everywhere

/// Maximum supported head dimension for attention kernels.
pub const MAX_HEAD_DIM: usize = 256;

/// Maximum supported block size for paged attention.
pub const MAX_BLOCK_SIZE: usize = 256;

/// Validate attention dimensions (batch_size, num_heads, seq_len, head_dim).
///
/// # Returns
/// - `Ok(())` if all dimensions are valid
/// - `Err(String)` describing the validation failure
///
/// # Example
/// ```ignore
/// validate_attention_dims(batch, heads, seq, head)?
///     .map_err(|e| MyError::InvalidConfig(e))?;
/// ```
#[inline]
pub fn validate_attention_dims(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(), String> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        ));
    }
    Ok(())
}

/// Validate paged attention dimensions with block_size.
///
/// # Returns
/// - `Ok(())` if all dimensions are valid
/// - `Err(String)` describing the validation failure
#[inline]
pub fn validate_paged_attention_dims(
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    seq_len: usize,
) -> Result<(), String> {
    if batch_size == 0 || num_heads == 0 || seq_len == 0 || head_dim == 0 || block_size == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if head_dim > MAX_HEAD_DIM {
        return Err(format!(
            "head_dim {} exceeds MAX_HEAD_DIM {}",
            head_dim, MAX_HEAD_DIM
        ));
    }
    if block_size > MAX_BLOCK_SIZE {
        return Err(format!(
            "block_size {} exceeds MAX_BLOCK_SIZE {}",
            block_size, MAX_BLOCK_SIZE
        ));
    }
    Ok(())
}

/// Validate embedding operation dimensions (binary/int8/int4 quantization).
///
/// # Arguments
/// - `dim` - Embedding dimension
/// - `alignment` - Required alignment (32 for binary, 4 for int8, 8 for int4)
/// - `num_queries` - Number of query vectors
/// - `num_vectors` - Number of database vectors
#[inline]
pub fn validate_embedding_dims(
    dim: usize,
    alignment: usize,
    num_queries: usize,
    num_vectors: usize,
) -> Result<(), String> {
    if dim == 0 || num_queries == 0 || num_vectors == 0 {
        return Err("Dimensions must be > 0".into());
    }
    if dim % alignment != 0 {
        return Err(format!(
            "dim {} must be multiple of {} for this operation",
            dim, alignment
        ));
    }
    Ok(())
}

/// Validate binary IP dimensions (requires dim % 32 == 0).
#[inline]
pub fn validate_binary_dim(dim: usize) -> Result<(), String> {
    if dim % 32 != 0 {
        return Err("dim must be multiple of 32 for binary IP".into());
    }
    Ok(())
}

/// Validate int8 dimensions (requires dim % 4 == 0).
#[inline]
pub fn validate_int8_dim(dim: usize) -> Result<(), String> {
    if dim % 4 != 0 {
        return Err("dim must be multiple of 4 for int8 dot product".into());
    }
    Ok(())
}

/// Validate int4 dimensions (requires dim % 8 == 0).
#[inline]
pub fn validate_int4_dim(dim: usize) -> Result<(), String> {
    if dim % 8 != 0 {
        return Err("dim must be multiple of 8 for int4 dot product".into());
    }
    Ok(())
}

/// Validate matryoshka truncation (target_dim <= full_dim).
#[inline]
pub fn validate_matryoshka_dims(full_dim: usize, target_dim: usize) -> Result<(), String> {
    if target_dim > full_dim {
        return Err(format!(
            "target_dim {} > full_dim {}",
            target_dim, full_dim
        ));
    }
    if target_dim == 0 || full_dim == 0 {
        return Err("Dimensions must be > 0".into());
    }
    Ok(())
}

/// Validate input buffer length matches expected.
#[inline]
pub fn validate_input_len(actual: usize, expected: usize, name: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!("{} len {} != expected {}", name, actual, expected));
    }
    Ok(())
}

/// Compute number of queries with overflow check.
///
/// # Returns
/// - `Ok(num_queries)` if computation succeeds
/// - `Err(String)` if overflow occurs
#[inline]
pub fn compute_num_queries(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) -> Result<usize, String> {
    batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(seq_len))
        .ok_or_else(|| "num_queries overflow".to_string())
}

/// Compute output length with overflow check.
///
/// # Returns
/// - `Ok(output_len)` if computation succeeds
/// - `Err(String)` if overflow occurs
#[inline]
pub fn compute_output_len(num_queries: usize, head_dim: usize) -> Result<usize, String> {
    num_queries
        .checked_mul(head_dim)
        .ok_or_else(|| "output_len overflow".to_string())
}

/// Validate dimensions fit in u32 (for GPU kernel parameters).
#[inline]
pub fn validate_u32_bounds(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(), String> {
    if batch_size > u32::MAX as usize {
        return Err("batch_size exceeds u32".into());
    }
    if num_heads > u32::MAX as usize {
        return Err("num_heads exceeds u32".into());
    }
    if seq_len > u32::MAX as usize {
        return Err("seq_len exceeds u32".into());
    }
    if head_dim > u32::MAX as usize {
        return Err("head_dim exceeds u32".into());
    }
    Ok(())
}

/// Validate dimensions fit in i32 (for CUDA/HIP kernel parameters).
#[inline]
pub fn validate_i32_bounds(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(), String> {
    if batch_size > i32::MAX as usize {
        return Err("batch_size exceeds i32".into());
    }
    if num_heads > i32::MAX as usize {
        return Err("num_heads exceeds i32".into());
    }
    if seq_len > i32::MAX as usize {
        return Err("seq_len exceeds i32".into());
    }
    if head_dim > i32::MAX as usize {
        return Err("head_dim exceeds i32".into());
    }
    Ok(())
}

/// Convert usize to u32 with error message.
#[inline]
pub fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{} exceeds u32", name))
}

/// Convert usize to i32 with error message.
#[inline]
pub fn to_i32(value: usize, name: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{} exceeds i32", name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_attention_dims_valid() {
        assert!(validate_attention_dims(1, 8, 512, 64).is_ok());
        assert!(validate_attention_dims(32, 32, 2048, 128).is_ok());
        assert!(validate_attention_dims(1, 1, 1, MAX_HEAD_DIM).is_ok());
    }

    #[test]
    fn test_validate_attention_dims_zero() {
        assert!(validate_attention_dims(0, 8, 512, 64).is_err());
        assert!(validate_attention_dims(1, 0, 512, 64).is_err());
        assert!(validate_attention_dims(1, 8, 0, 64).is_err());
        assert!(validate_attention_dims(1, 8, 512, 0).is_err());
    }

    #[test]
    fn test_validate_attention_dims_head_dim_exceeded() {
        assert!(validate_attention_dims(1, 8, 512, MAX_HEAD_DIM + 1).is_err());
    }

    #[test]
    fn test_validate_binary_dim() {
        assert!(validate_binary_dim(32).is_ok());
        assert!(validate_binary_dim(64).is_ok());
        assert!(validate_binary_dim(1024).is_ok());
        assert!(validate_binary_dim(31).is_err());
        assert!(validate_binary_dim(33).is_err());
    }

    #[test]
    fn test_validate_int8_dim() {
        assert!(validate_int8_dim(4).is_ok());
        assert!(validate_int8_dim(128).is_ok());
        assert!(validate_int8_dim(3).is_err());
        assert!(validate_int8_dim(5).is_err());
    }

    #[test]
    fn test_validate_int4_dim() {
        assert!(validate_int4_dim(8).is_ok());
        assert!(validate_int4_dim(256).is_ok());
        assert!(validate_int4_dim(7).is_err());
        assert!(validate_int4_dim(9).is_err());
    }

    #[test]
    fn test_compute_num_queries() {
        assert_eq!(compute_num_queries(2, 8, 512).unwrap(), 8192);
        assert!(compute_num_queries(usize::MAX, 2, 2).is_err());
    }

    #[test]
    fn test_validate_matryoshka_dims() {
        assert!(validate_matryoshka_dims(1024, 512).is_ok());
        assert!(validate_matryoshka_dims(1024, 1024).is_ok());
        assert!(validate_matryoshka_dims(512, 1024).is_err());
        assert!(validate_matryoshka_dims(0, 512).is_err());
    }
}
