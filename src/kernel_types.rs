use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedBits {
    Int4,
    Int2,
    Int1,
}

impl PackedBits {
    pub const fn bits(self) -> usize {
        match self {
            PackedBits::Int4 => 4,
            PackedBits::Int2 => 2,
            PackedBits::Int1 => 1,
        }
    }

    pub const fn values_per_byte(self) -> usize {
        8 / self.bits()
    }

    pub const fn from_bits(bits: usize) -> Option<Self> {
        match bits {
            4 => Some(PackedBits::Int4),
            2 => Some(PackedBits::Int2),
            1 => Some(PackedBits::Int1),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U8,
    PackedU8(PackedBits),
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::PackedU8(_) => 1,
        }
    }

    pub const fn from_size_bytes(size: usize) -> Option<Self> {
        match size {
            4 => Some(DType::F32),
            2 => Some(DType::F16),
            1 => Some(DType::U8),
            _ => None,
        }
    }

    pub const fn bits_per_value(self) -> usize {
        match self {
            DType::F32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::U8 => 8,
            DType::PackedU8(bits) => bits.bits(),
        }
    }

    pub const fn values_per_byte(self) -> usize {
        match self {
            DType::PackedU8(bits) => bits.values_per_byte(),
            _ => 1,
        }
    }

    pub const fn is_packed(self) -> bool {
        matches!(self, DType::PackedU8(_))
    }

    pub const fn storage_bytes_for(self, values: usize) -> Option<usize> {
        if values == 0 {
            return Some(0);
        }
        if self.is_packed() {
            let per = self.values_per_byte();
            if per == 0 {
                return None;
            }
            return match values.checked_add(per - 1) {
                Some(total) => Some(total / per),
                None => None,
            };
        }
        values.checked_mul(self.size_bytes())
    }
}

/// 页面状态枚举 (用于 Swap-in/Swap-out)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// 在 GPU 内存中且正在使用
    Active,
    /// 在 GPU 内存中但未使用 (可换出)
    Standby,
    /// 已换出到 CPU 内存
    Swapped,
}

/// Swap 配置
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwapConfig {
    /// 是否启用 swap
    pub enable_swap: bool,
    /// CPU 保留内存 (MB)
    pub cpu_reserve_mb: usize,
    /// GPU 使用率阈值 (0.0-1.0)，超过此值时触发 swap-out
    pub swap_threshold: f32,
    /// LRU 粒度 (页数)
    pub lru_granularity: usize,
}

impl Default for SwapConfig {
    fn default() -> Self {
        Self {
            enable_swap: true,
            cpu_reserve_mb: 512,
            swap_threshold: 0.85,
            lru_granularity: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype_size: usize,
    pub page_size: usize,
    /// Swap 配置 (可选)
    pub swap_config: Option<SwapConfig>,
}

impl KvCacheConfig {
    pub fn is_paged(&self) -> bool {
        self.page_size > 0 && self.page_size < self.max_seq_len
    }

    pub fn effective_page_size(&self) -> usize {
        if self.page_size == 0 || self.page_size >= self.max_seq_len {
            self.max_seq_len
        } else {
            self.page_size
        }
    }

    pub fn pages_per_layer(&self) -> Option<usize> {
        let page = self.effective_page_size();
        if page == 0 {
            return None;
        }
        let pages = (self.max_seq_len + page - 1) / page;
        Some(pages.max(1))
    }

    pub fn kv_stride(&self) -> Option<usize> {
        self.num_heads.checked_mul(self.head_dim)
    }

    pub fn total_bytes(&self) -> Option<usize> {
        let elems_per_token = self.kv_stride()?;
        if self.is_paged() {
            let page = self.effective_page_size();
            let pages_per_layer = self.pages_per_layer()?;
            let per_page = page.checked_mul(elems_per_token)?.checked_mul(2)?;
            let per_layer = per_page.checked_mul(pages_per_layer)?;
            let total = per_layer.checked_mul(self.num_layers)?;
            return total.checked_mul(self.dtype_size);
        }
        let per_layer = elems_per_token.checked_mul(self.max_seq_len)?;
        let total_tokens = per_layer.checked_mul(self.num_layers)?;
        let total_kv = total_tokens.checked_mul(2)?;
        total_kv.checked_mul(self.dtype_size)
    }

    pub fn dtype(&self) -> Option<DType> {
        DType::from_size_bytes(self.dtype_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEncoding {
    Rope,
    Alibi,
}

impl Default for PositionEncoding {
    fn default() -> Self {
        PositionEncoding::Rope
    }
}

pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    if num_heads == 0 {
        return Vec::new();
    }

    fn slopes_power_of_two(n: usize) -> Vec<f32> {
        let exponent = (n as f32).log2();
        let start = 2f32.powf(-(2f32.powf(-(exponent - 3.0))));
        let ratio = start;
        let mut slopes = Vec::with_capacity(n);
        for i in 0..n {
            slopes.push(start * ratio.powi(i as i32));
        }
        slopes
    }

    let mut pow2 = 1usize;
    while pow2.saturating_mul(2) <= num_heads {
        pow2 = pow2.saturating_mul(2);
    }

    if pow2 == num_heads {
        return slopes_power_of_two(num_heads);
    }

    let mut slopes = slopes_power_of_two(pow2);
    let extended = slopes_power_of_two(pow2.saturating_mul(2));
    for idx in (0..extended.len()).step_by(2) {
        if slopes.len() == num_heads {
            break;
        }
        slopes.push(extended[idx]);
    }
    slopes.truncate(num_heads);
    slopes
}

pub fn precompute_rope_tables(
    max_seq_len: usize,
    rotary_dim: usize,
    base: f32,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    if max_seq_len == 0 || half == 0 {
        return (Vec::new(), Vec::new());
    }
    let total = max_seq_len.saturating_mul(half);
    let mut cos_table = vec![0.0f32; total];
    let mut sin_table = vec![0.0f32; total];
    let rotary = rotary_dim as f32;
    for pos in 0..max_seq_len {
        let pos_f = pos as f32;
        for idx in 0..half {
            let freq = base.powf(-2.0 * idx as f32 / rotary);
            let angle = pos_f * scale * freq;
            let offset = pos * half + idx;
            cos_table[offset] = angle.cos();
            sin_table[offset] = angle.sin();
        }
    }
    (cos_table, sin_table)
}

pub fn f16_to_f32_into(input: &[f16], output: &mut [f32]) -> Result<(), &'static str> {
    if output.len() < input.len() {
        return Err("f16_to_f32 output too small");
    }
    for (dst, src) in output.iter_mut().zip(input.iter()) {
        *dst = src.to_f32();
    }
    Ok(())
}

pub fn bf16_to_f32_into(input: &[bf16], output: &mut [f32]) -> Result<(), &'static str> {
    if output.len() < input.len() {
        return Err("bf16_to_f32 output too small");
    }
    for (dst, src) in output.iter_mut().zip(input.iter()) {
        *dst = src.to_f32();
    }
    Ok(())
}

pub fn f32_to_f16_into(input: &[f32], output: &mut [f16]) -> Result<(), &'static str> {
    if output.len() < input.len() {
        return Err("f32_to_f16 output too small");
    }
    for (dst, src) in output.iter_mut().zip(input.iter()) {
        *dst = f16::from_f32(*src);
    }
    Ok(())
}

pub fn f32_to_bf16_into(input: &[f32], output: &mut [bf16]) -> Result<(), &'static str> {
    if output.len() < input.len() {
        return Err("f32_to_bf16 output too small");
    }
    for (dst, src) in output.iter_mut().zip(input.iter()) {
        *dst = bf16::from_f32(*src);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::PackedU8(PackedBits::Int4).size_bytes(), 1);
    }

    #[test]
    fn dtype_storage_and_bits() {
        assert_eq!(PackedBits::from_bits(4), Some(PackedBits::Int4));
        assert_eq!(PackedBits::from_bits(2), Some(PackedBits::Int2));
        assert_eq!(PackedBits::from_bits(1), Some(PackedBits::Int1));
        assert_eq!(PackedBits::from_bits(3), None);

        let dtype = DType::PackedU8(PackedBits::Int4);
        assert_eq!(dtype.bits_per_value(), 4);
        assert_eq!(dtype.values_per_byte(), 2);
        assert_eq!(dtype.storage_bytes_for(0), Some(0));
        assert_eq!(dtype.storage_bytes_for(1), Some(1));
        assert_eq!(dtype.storage_bytes_for(2), Some(1));
        assert_eq!(dtype.storage_bytes_for(3), Some(2));

        let dtype = DType::PackedU8(PackedBits::Int2);
        assert_eq!(dtype.bits_per_value(), 2);
        assert_eq!(dtype.values_per_byte(), 4);
        assert_eq!(dtype.storage_bytes_for(4), Some(1));
        assert_eq!(dtype.storage_bytes_for(5), Some(2));

        let dtype = DType::PackedU8(PackedBits::Int1);
        assert_eq!(dtype.bits_per_value(), 1);
        assert_eq!(dtype.values_per_byte(), 8);
        assert_eq!(dtype.storage_bytes_for(8), Some(1));
        assert_eq!(dtype.storage_bytes_for(9), Some(2));
    }

    #[test]
    fn f16_conversion_roundtrip() {
        let input = vec![0.0f32, 1.25, -2.5, 3.75];
        let mut half_buf = vec![f16::from_f32(0.0); input.len()];
        f32_to_f16_into(&input, &mut half_buf).unwrap();
        let mut output = vec![0.0f32; input.len()];
        f16_to_f32_into(&half_buf, &mut output).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn bf16_conversion_roundtrip() {
        let input = vec![0.0f32, 1.25, -2.5, 3.75];
        let mut bf_buf = vec![bf16::from_f32(0.0); input.len()];
        f32_to_bf16_into(&input, &mut bf_buf).unwrap();
        let mut output = vec![0.0f32; input.len()];
        bf16_to_f32_into(&bf_buf, &mut output).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-2);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratorForwardConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub rope_interleaved: bool,
    pub rope_precompute: bool,
    pub position_encoding: PositionEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}
