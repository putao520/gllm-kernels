//! TileOps trait — abstract interface for matrix tile accelerators (AMX, SME).
//!
//! Provides a hardware-agnostic API for tile-based matrix operations.
//! Currently implemented for Intel AMX; ARM SME can be added in the future.

use super::simd_ops::BaseReg;

/// Virtual tile register (0..7 for AMX, 0..31 for SME).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TReg(pub u8);

/// Tile configuration descriptor.
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Number of rows in the tile.
    pub rows: u8,
    /// Number of columns in bytes.
    pub cols_bytes: u16,
}

/// Tile accelerator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileAccelKind {
    /// Intel AMX (SPR+)
    Amx,
    /// ARM SME (future)
    Sme,
}

/// Supported tile data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileDtype {
    BF16,
    INT8,
    FP16,
}

/// Platform-agnostic tile matrix operation interface.
///
/// Backends implement this trait to emit tile instructions. The algorithm
/// layer calls these methods for GEMM acceleration without knowing the
/// specific tile ISA.
pub trait TileOps {
    /// Configure tile register layouts.
    ///
    /// Must be called before any tile load/store/compute operations.
    /// On AMX this emits LDTILECFG.
    fn tile_configure(&mut self, configs: &[(TReg, TileConfig)]) -> Result<(), String>;

    /// Release all tile state.
    ///
    /// Must be called after tile operations are complete.
    /// On AMX this emits TILERELEASE.
    fn tile_release(&mut self) -> Result<(), String>;

    /// Zero a tile accumulator register.
    fn tile_zero(&mut self, dst: TReg) -> Result<(), String>;

    /// Load a tile from memory.
    ///
    /// `base` is the memory base pointer, `stride` is the row stride in bytes.
    fn tile_load(&mut self, dst: TReg, base: BaseReg, stride: BaseReg) -> Result<(), String>;

    /// Store a tile to memory.
    ///
    /// `base` is the memory base pointer, `stride` is the row stride in bytes.
    fn tile_store(&mut self, base: BaseReg, stride: BaseReg, src: TReg) -> Result<(), String>;

    /// BF16 tile matrix multiply-accumulate: dst += a * b
    ///
    /// On AMX this emits TDPBF16PS.
    fn tile_dpbf16(&mut self, dst: TReg, a: TReg, b: TReg) -> Result<(), String>;

    /// INT8 signed×signed tile matrix multiply-accumulate: dst += a * b
    ///
    /// On AMX this emits TDPBSSD.
    fn tile_dpbssd(&mut self, dst: TReg, a: TReg, b: TReg) -> Result<(), String>;

    /// Maximum number of rows per tile.
    fn tile_max_rows(&self) -> u8;

    /// Maximum number of column bytes per tile.
    fn tile_max_cols_bytes(&self) -> u16;

    /// Number of tile registers available.
    fn tile_count(&self) -> u8;

    /// Whether this backend supports tile operations.
    fn has_tile_ops(&self) -> bool;

    /// What kind of tile accelerator is available (if any).
    fn tile_accel_kind(&self) -> Option<TileAccelKind>;

    /// Supported data types for tile operations.
    fn tile_supported_dtypes(&self) -> Vec<TileDtype>;
}
