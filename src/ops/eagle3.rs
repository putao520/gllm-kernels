//! EAGLE-3 adaptive draft length speculative decoding.
//!
//! Based on EAGLE-3 (NeurIPS'25): 2-6x inference acceleration through
//! multi-layer feature fusion, token-level confidence prediction, and
//! adaptive draft length scheduling.
//!
//! # Key Features
//! - Multi-layer feature fusion (vs EAGLE-2 single layer)
//! - Token-level confidence prediction (vs sequence-level)
//! - Adaptive draft length based on acceptance history
//! - Training-time test distribution simulation

#[path = "eagle3/config.rs"]
mod config;
#[path = "eagle3/decoder.rs"]
mod decoder;
#[path = "eagle3/fuse.rs"]
mod fuse;
#[path = "eagle3/predictor.rs"]
mod predictor;
#[path = "eagle3/scheduler.rs"]
mod scheduler;
#[path = "eagle3/types.rs"]
mod types;

pub use config::AdaptiveDraftConfig;
pub use decoder::Eagle3Decoder;
pub use fuse::{fuse_multi_layer_hidden, predict_confidence};
pub use predictor::ConfidencePredictor;
pub use scheduler::LengthScheduler;
pub use types::{Eagle3Draft, Eagle3DraftToken, Eagle3Stats, Eagle3Verification};

#[cfg(all(test, feature = "cpu"))]
#[path = "eagle3/tests.rs"]
mod tests;
