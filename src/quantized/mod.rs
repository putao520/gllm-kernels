use half::f16;

/// Q4_0 quantized block: 32 values packed into 16 bytes with a per-block scale.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4_0Block {
    pub scale: f16,
    pub qs: [u8; 16],
}

/// Q8_0 quantized block: 32 int8 values with a per-block scale.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Q8_0Block {
    pub scale: f16,
    pub qs: [i8; 32],
}

/// AWQ INT4 packed weights and per-group scales.
#[derive(Clone, Debug)]
pub struct AwqWeight {
    pub qweight: Vec<u32>,
    pub qzeros: Vec<u32>,
    pub scales: Vec<f16>,
    pub group_size: usize,
}
