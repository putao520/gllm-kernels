//! # Chunked Prefill / POD-Attention (REQ-OP-015)
//!
//! Implements chunked prefill with prefill-decode overlap for long prompt scenarios.
//!
//! ## Features
//! - **Chunked Prefill**: Split long prompts into chunks for incremental processing
//! - **POD-Attention**: Prefill-On-Demand with dynamic SM allocation
//! - **Interleaved Execution**: Mix prefill chunks with decode operations
//! - **Memory Efficiency**: Chunk-level KV management
//!
//! ## References
//! - POD-Attention (ASPLOS'25): 22% throughput improvement
//! - FlashInfer: Customizable Attention Engine
//! - Sarathi: Chunked Prefill with pipelining
//!
//! ## SPEC Compliance
//! - ARCH-OP-015: ChunkConfig, PODAttentionConfig, ChunkedPrefillScheduler
//! - Target: 10-22% throughput improvement

use std::collections::VecDeque;
use std::time::Instant;

/// Request identifier
pub type RequestId = u64;

/// Token type
pub type TokenId = u32;

// ============================================================================
// Request Types
// ============================================================================

/// A prefill request (initial prompt processing)
#[derive(Debug, Clone)]
pub struct PrefillRequest {
    /// Unique request ID
    pub request_id: RequestId,
    /// Input tokens to process
    pub tokens: Vec<TokenId>,
    /// Starting position (for continuation)
    pub start_pos: usize,
    /// Priority (higher = more urgent)
    pub priority: u32,
    /// Creation timestamp
    pub created_at: Instant,
}

impl PrefillRequest {
    /// Create new prefill request
    pub fn new(request_id: RequestId, tokens: Vec<TokenId>) -> Self {
        Self {
            request_id,
            tokens,
            start_pos: 0,
            priority: 0,
            created_at: Instant::now(),
        }
    }

    /// Create with priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Get total token count
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Get remaining tokens to process
    pub fn remaining_tokens(&self) -> usize {
        self.tokens.len().saturating_sub(self.start_pos)
    }

    /// Check if request is complete
    pub fn is_complete(&self) -> bool {
        self.start_pos >= self.tokens.len()
    }

    /// Get waiting time
    pub fn wait_time(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

/// A decode request (autoregressive generation)
#[derive(Debug, Clone)]
pub struct DecodeRequest {
    /// Unique request ID
    pub request_id: RequestId,
    /// Current KV cache length
    pub kv_length: usize,
    /// Token to decode
    pub token: TokenId,
    /// Priority
    pub priority: u32,
    /// Batch index (for grouping)
    pub batch_idx: usize,
}

impl DecodeRequest {
    /// Create new decode request
    pub fn new(request_id: RequestId, token: TokenId, kv_length: usize) -> Self {
        Self {
            request_id,
            kv_length,
            token,
            priority: 0,
            batch_idx: 0,
        }
    }

    /// Create with batch index
    pub fn with_batch_idx(mut self, idx: usize) -> Self {
        self.batch_idx = idx;
        self
    }
}

// ============================================================================
// Chunk Config (ARCH-OP-015)
// ============================================================================

/// Configuration for chunked prefill
///
/// SPEC: ARCH-OP-015 ChunkConfig structure
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Tokens per chunk
    pub chunk_size: usize,
    /// Maximum chunks per batch
    pub max_chunks_per_batch: usize,
    /// Interleave with decode operations
    pub interleave_decodes: bool,
    /// Dynamic chunk size based on prompt length
    pub dynamic_chunk_size: bool,
    /// Minimum chunk size (for dynamic)
    pub min_chunk_size: usize,
    /// Maximum chunk size (for dynamic)
    pub max_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 2048,
            max_chunks_per_batch: 4,
            interleave_decodes: true,
            dynamic_chunk_size: true,
            min_chunk_size: 512,
            max_chunk_size: 4096,
        }
    }
}

impl ChunkConfig {
    /// Create config for short prompts
    pub fn for_short_prompts() -> Self {
        Self {
            chunk_size: 1024,
            max_chunks_per_batch: 8,
            ..Default::default()
        }
    }

    /// Create config for long prompts
    pub fn for_long_prompts() -> Self {
        Self {
            chunk_size: 4096,
            max_chunks_per_batch: 2,
            dynamic_chunk_size: false,
            ..Default::default()
        }
    }

    /// Calculate optimal chunk size for prompt length
    pub fn optimal_chunk_size(&self, prompt_len: usize) -> usize {
        if !self.dynamic_chunk_size {
            return self.chunk_size;
        }

        // Heuristic: aim for 4-8 chunks for long prompts
        let target_chunks = if prompt_len > 8192 { 8 } else { 4 };
        let computed = prompt_len / target_chunks;

        computed.clamp(self.min_chunk_size, self.max_chunk_size)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_size == 0 {
            return Err("chunk_size must be > 0".to_string());
        }
        if self.max_chunks_per_batch == 0 {
            return Err("max_chunks_per_batch must be > 0".to_string());
        }
        if self.min_chunk_size > self.max_chunk_size {
            return Err("min_chunk_size must be <= max_chunk_size".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// POD Attention Config (ARCH-OP-015)
// ============================================================================

/// Configuration for POD (Prefill-On-Demand) Attention
///
/// SPEC: ARCH-OP-015 PODAttentionConfig structure
#[derive(Debug, Clone)]
pub struct PODAttentionConfig {
    /// SM ratio allocated to prefill operations
    pub prefill_sm_ratio: f32,
    /// SM ratio allocated to decode operations
    pub decode_sm_ratio: f32,
    /// Enable dynamic SM allocation
    pub enable_dynamic_allocation: bool,
    /// Minimum SMs per task
    pub min_sm_per_task: usize,
    /// Total available SMs (device-specific)
    pub total_sms: usize,
}

impl Default for PODAttentionConfig {
    fn default() -> Self {
        Self {
            prefill_sm_ratio: 0.6,
            decode_sm_ratio: 0.4,
            enable_dynamic_allocation: true,
            min_sm_per_task: 4,
            total_sms: 108, // A100 default
        }
    }
}

impl PODAttentionConfig {
    /// Create config for prefill-heavy workloads
    pub fn prefill_heavy() -> Self {
        Self {
            prefill_sm_ratio: 0.8,
            decode_sm_ratio: 0.2,
            ..Default::default()
        }
    }

    /// Create config for decode-heavy workloads
    pub fn decode_heavy() -> Self {
        Self {
            prefill_sm_ratio: 0.3,
            decode_sm_ratio: 0.7,
            ..Default::default()
        }
    }

    /// Create config for specific GPU
    pub fn for_gpu(total_sms: usize) -> Self {
        Self {
            total_sms,
            ..Default::default()
        }
    }

    /// Get SM allocation for prefill
    pub fn prefill_sms(&self) -> usize {
        ((self.total_sms as f32 * self.prefill_sm_ratio) as usize).max(self.min_sm_per_task)
    }

    /// Get SM allocation for decode
    pub fn decode_sms(&self) -> usize {
        ((self.total_sms as f32 * self.decode_sm_ratio) as usize).max(self.min_sm_per_task)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.prefill_sm_ratio + self.decode_sm_ratio > 1.0 {
            return Err("SM ratios must sum to <= 1.0".to_string());
        }
        if self.prefill_sm_ratio < 0.0 || self.decode_sm_ratio < 0.0 {
            return Err("SM ratios must be >= 0".to_string());
        }
        if self.min_sm_per_task == 0 {
            return Err("min_sm_per_task must be > 0".to_string());
        }
        if self.total_sms < self.min_sm_per_task * 2 {
            return Err("total_sms must be >= 2 * min_sm_per_task".to_string());
        }
        Ok(())
    }

    /// Dynamically adjust ratios based on queue depths
    pub fn adjust_for_load(&mut self, prefill_queue_len: usize, decode_queue_len: usize) {
        if !self.enable_dynamic_allocation {
            return;
        }

        let total = prefill_queue_len + decode_queue_len;
        if total == 0 {
            return;
        }

        // Adjust ratios based on queue pressure
        let prefill_pressure = prefill_queue_len as f32 / total as f32;
        let decode_pressure = decode_queue_len as f32 / total as f32;

        // Smooth adjustment (move 10% toward target)
        let target_prefill = prefill_pressure * 0.8 + 0.1; // Bias toward prefill
        let target_decode = decode_pressure * 0.8 + 0.1;

        self.prefill_sm_ratio = self.prefill_sm_ratio * 0.9 + target_prefill * 0.1;
        self.decode_sm_ratio = self.decode_sm_ratio * 0.9 + target_decode * 0.1;

        // Normalize
        let sum = self.prefill_sm_ratio + self.decode_sm_ratio;
        if sum > 0.0 {
            self.prefill_sm_ratio /= sum;
            self.decode_sm_ratio /= sum;
        }
    }
}

// ============================================================================
// Scheduled Batch
// ============================================================================

/// A scheduled batch of mixed prefill and decode operations
#[derive(Debug)]
pub struct ScheduledBatch {
    /// Prefill chunks in this batch
    pub prefill_chunks: Vec<PrefillChunk>,
    /// Decode requests in this batch
    pub decode_requests: Vec<DecodeRequest>,
    /// SM allocation for prefill
    pub prefill_sms: usize,
    /// SM allocation for decode
    pub decode_sms: usize,
    /// Batch ID
    pub batch_id: u64,
    /// Scheduling timestamp
    pub scheduled_at: Instant,
}

/// A chunk of a prefill request
#[derive(Debug)]
pub struct PrefillChunk {
    /// Source request ID
    pub request_id: RequestId,
    /// Tokens in this chunk
    pub tokens: Vec<TokenId>,
    /// Start position in original request
    pub start_pos: usize,
    /// End position in original request
    pub end_pos: usize,
    /// Chunk index (0-based)
    pub chunk_idx: usize,
    /// Total chunks for this request
    pub total_chunks: usize,
    /// Is this the last chunk?
    pub is_last: bool,
}

impl ScheduledBatch {
    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.prefill_chunks.is_empty() && self.decode_requests.is_empty()
    }

    /// Get total tokens in batch
    pub fn total_tokens(&self) -> usize {
        let prefill_tokens: usize = self.prefill_chunks.iter().map(|c| c.tokens.len()).sum();
        let decode_tokens = self.decode_requests.len();
        prefill_tokens + decode_tokens
    }

    /// Get prefill token count
    pub fn prefill_tokens(&self) -> usize {
        self.prefill_chunks.iter().map(|c| c.tokens.len()).sum()
    }

    /// Get decode token count
    pub fn decode_tokens(&self) -> usize {
        self.decode_requests.len()
    }
}

// ============================================================================
// Batch Output
// ============================================================================

/// Output from executing a batch
#[derive(Debug)]
pub struct BatchOutput {
    /// Prefill outputs (one per chunk)
    pub prefill_outputs: Vec<PrefillOutput>,
    /// Decode outputs (one per request)
    pub decode_outputs: Vec<DecodeOutput>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// Output from a prefill chunk
#[derive(Debug)]
pub struct PrefillOutput {
    /// Request ID
    pub request_id: RequestId,
    /// Hidden states [chunk_size, hidden_dim]
    pub hidden_states: Vec<f32>,
    /// KV cache entries created
    pub kv_cache_entries: usize,
    /// Chunk completed
    pub chunk_idx: usize,
}

/// Output from a decode operation
#[derive(Debug)]
pub struct DecodeOutput {
    /// Request ID
    pub request_id: RequestId,
    /// Next token logits [vocab_size]
    pub logits: Vec<f32>,
    /// Predicted token (argmax)
    pub predicted_token: TokenId,
}

// ============================================================================
// Chunked Prefill Scheduler (ARCH-OP-015)
// ============================================================================

/// Scheduler for chunked prefill with POD-Attention
///
/// SPEC: ARCH-OP-015 ChunkedPrefillScheduler structure
pub struct ChunkedPrefillScheduler {
    /// Prefill request queue
    prefill_queue: VecDeque<PrefillRequest>,
    /// Decode request queue
    decode_queue: VecDeque<DecodeRequest>,
    /// Chunk configuration
    chunk_config: ChunkConfig,
    /// POD-Attention configuration
    pod_config: PODAttentionConfig,
    /// Next batch ID
    next_batch_id: u64,
    /// Statistics
    stats: SchedulerStats,
}

impl ChunkedPrefillScheduler {
    /// Create new scheduler
    pub fn new(chunk_config: ChunkConfig, pod_config: PODAttentionConfig) -> Result<Self, String> {
        chunk_config.validate()?;
        pod_config.validate()?;

        Ok(Self {
            prefill_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            chunk_config,
            pod_config,
            next_batch_id: 1,
            stats: SchedulerStats::default(),
        })
    }

    /// Submit a prefill request
    pub fn submit_prefill(&mut self, request: PrefillRequest) -> RequestId {
        let id = request.request_id;
        self.stats.prefill_submitted += 1;

        // Insert by priority (higher priority first)
        let pos = self
            .prefill_queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.prefill_queue.len());

        self.prefill_queue.insert(pos, request);
        id
    }

    /// Submit a decode request
    pub fn submit_decode(&mut self, request: DecodeRequest) -> RequestId {
        let id = request.request_id;
        self.stats.decode_submitted += 1;
        self.decode_queue.push_back(request);
        id
    }

    /// Schedule next batch
    pub fn schedule_batch(&mut self) -> ScheduledBatch {
        // Dynamically adjust SM allocation based on queue pressure
        self.pod_config.adjust_for_load(self.prefill_queue.len(), self.decode_queue.len());

        let mut prefill_chunks = Vec::new();
        let mut decode_requests = Vec::new();

        // Schedule prefill chunks
        if !self.prefill_queue.is_empty() {
            let mut chunks_scheduled = 0;

            while chunks_scheduled < self.chunk_config.max_chunks_per_batch
                && !self.prefill_queue.is_empty()
            {
                if let Some(mut request) = self.prefill_queue.pop_front() {
                    let chunk = self.create_chunk(&mut request);
                    prefill_chunks.push(chunk);
                    chunks_scheduled += 1;

                    // If request has more tokens, put it back
                    if !request.is_complete() {
                        self.prefill_queue.push_front(request);
                    } else {
                        self.stats.prefill_completed += 1;
                    }
                }
            }
        }

        // Schedule decode requests (interleaved)
        if self.chunk_config.interleave_decodes {
            // Take all pending decode requests
            while let Some(request) = self.decode_queue.pop_front() {
                decode_requests.push(request);
            }
            self.stats.decode_completed += decode_requests.len() as u64;
        }

        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        ScheduledBatch {
            prefill_chunks,
            decode_requests,
            prefill_sms: self.pod_config.prefill_sms(),
            decode_sms: self.pod_config.decode_sms(),
            batch_id,
            scheduled_at: Instant::now(),
        }
    }

    /// Execute a scheduled batch (CPU reference implementation)
    pub fn execute_batch(&self, batch: &ScheduledBatch) -> BatchOutput {
        let start = Instant::now();

        // Execute prefill chunks
        let prefill_outputs: Vec<PrefillOutput> = batch
            .prefill_chunks
            .iter()
            .map(|chunk| {
                // Simulate prefill computation
                let hidden_dim = 4096; // Typical hidden dimension
                let hidden_states = vec![0.0f32; chunk.tokens.len() * hidden_dim];

                PrefillOutput {
                    request_id: chunk.request_id,
                    hidden_states,
                    kv_cache_entries: chunk.tokens.len(),
                    chunk_idx: chunk.chunk_idx,
                }
            })
            .collect();

        // Execute decode requests
        let decode_outputs: Vec<DecodeOutput> = batch
            .decode_requests
            .iter()
            .map(|request| {
                // Simulate decode computation
                let vocab_size = 32000;
                let logits = vec![0.0f32; vocab_size];

                DecodeOutput {
                    request_id: request.request_id,
                    logits,
                    predicted_token: 0, // Would be argmax of logits
                }
            })
            .collect();

        let execution_time_us = start.elapsed().as_micros() as u64;

        BatchOutput {
            prefill_outputs,
            decode_outputs,
            execution_time_us,
        }
    }

    /// Get current queue depths
    pub fn queue_depths(&self) -> (usize, usize) {
        (self.prefill_queue.len(), self.decode_queue.len())
    }

    /// Get statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Check if there's work to do
    pub fn has_work(&self) -> bool {
        !self.prefill_queue.is_empty() || !self.decode_queue.is_empty()
    }

    /// Clear all queues
    pub fn clear(&mut self) {
        self.prefill_queue.clear();
        self.decode_queue.clear();
    }

    /// Get current configuration
    pub fn chunk_config(&self) -> &ChunkConfig {
        &self.chunk_config
    }

    /// Get POD configuration
    pub fn pod_config(&self) -> &PODAttentionConfig {
        &self.pod_config
    }

    // Private helpers

    fn create_chunk(&self, request: &mut PrefillRequest) -> PrefillChunk {
        let prompt_len = request.tokens.len();
        let chunk_size = self.chunk_config.optimal_chunk_size(prompt_len);
        let total_chunks = (prompt_len + chunk_size - 1) / chunk_size;

        let start_pos = request.start_pos;
        let end_pos = (start_pos + chunk_size).min(prompt_len);

        let tokens = request.tokens[start_pos..end_pos].to_vec();
        let chunk_idx = start_pos / chunk_size;
        let is_last = end_pos >= prompt_len;

        // Advance request position
        request.start_pos = end_pos;

        PrefillChunk {
            request_id: request.request_id,
            tokens,
            start_pos,
            end_pos,
            chunk_idx,
            total_chunks,
            is_last,
        }
    }
}

// ============================================================================
// Scheduler Statistics
// ============================================================================

/// Statistics for the scheduler
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total prefill requests submitted
    pub prefill_submitted: u64,
    /// Total prefill requests completed
    pub prefill_completed: u64,
    /// Total decode requests submitted
    pub decode_submitted: u64,
    /// Total decode requests completed
    pub decode_completed: u64,
    /// Total batches scheduled
    pub batches_scheduled: u64,
    /// Total tokens processed
    pub tokens_processed: u64,
}

impl SchedulerStats {
    /// Calculate prefill completion rate
    pub fn prefill_completion_rate(&self) -> f64 {
        if self.prefill_submitted == 0 {
            0.0
        } else {
            self.prefill_completed as f64 / self.prefill_submitted as f64
        }
    }

    /// Calculate decode completion rate
    pub fn decode_completion_rate(&self) -> f64 {
        if self.decode_submitted == 0 {
            0.0
        } else {
            self.decode_completed as f64 / self.decode_submitted as f64
        }
    }

    /// Calculate average tokens per batch
    pub fn avg_tokens_per_batch(&self) -> f64 {
        if self.batches_scheduled == 0 {
            0.0
        } else {
            self.tokens_processed as f64 / self.batches_scheduled as f64
        }
    }

    /// Estimate throughput improvement from chunking
    pub fn estimated_throughput_improvement(&self) -> f32 {
        // Based on interleaving efficiency
        let total_ops = self.prefill_completed + self.decode_completed;
        if total_ops == 0 {
            return 1.0;
        }

        let interleave_ratio = self.decode_completed as f32 / total_ops as f32;
        // Interleaving can provide up to 22% improvement
        1.0 + (interleave_ratio * 0.22)
    }
}

// ============================================================================
// Chunk Iterator
// ============================================================================

/// Iterator over chunks of a token sequence
pub struct ChunkIterator<'a> {
    tokens: &'a [TokenId],
    chunk_size: usize,
    position: usize,
}

impl<'a> ChunkIterator<'a> {
    /// Create new chunk iterator
    pub fn new(tokens: &'a [TokenId], chunk_size: usize) -> Self {
        Self {
            tokens,
            chunk_size,
            position: 0,
        }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = &'a [TokenId];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.tokens.len() {
            return None;
        }

        let start = self.position;
        let end = (start + self.chunk_size).min(self.tokens.len());
        self.position = end;

        Some(&self.tokens[start..end])
    }
}

/// Helper function to iterate over chunks
pub fn chunk_tokens(tokens: &[TokenId], chunk_size: usize) -> ChunkIterator<'_> {
    ChunkIterator::new(tokens, chunk_size)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_request() {
        let tokens: Vec<TokenId> = (0..1000).collect();
        let mut request = PrefillRequest::new(1, tokens);

        assert_eq!(request.token_count(), 1000);
        assert_eq!(request.remaining_tokens(), 1000);
        assert!(!request.is_complete());

        request.start_pos = 500;
        assert_eq!(request.remaining_tokens(), 500);

        request.start_pos = 1000;
        assert!(request.is_complete());
    }

    #[test]
    fn test_chunk_config_validation() {
        let valid = ChunkConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = ChunkConfig {
            chunk_size: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid2 = ChunkConfig {
            min_chunk_size: 2000,
            max_chunk_size: 1000,
            ..Default::default()
        };
        assert!(invalid2.validate().is_err());
    }

    #[test]
    fn test_chunk_config_optimal_size() {
        let config = ChunkConfig::default();

        // Short prompt
        let size = config.optimal_chunk_size(1000);
        assert!(size >= config.min_chunk_size);
        assert!(size <= config.max_chunk_size);

        // Long prompt
        let size = config.optimal_chunk_size(16000);
        assert!(size >= config.min_chunk_size);
        assert!(size <= config.max_chunk_size);

        // Non-dynamic
        let static_config = ChunkConfig {
            dynamic_chunk_size: false,
            chunk_size: 1024,
            ..Default::default()
        };
        assert_eq!(static_config.optimal_chunk_size(16000), 1024);
    }

    #[test]
    fn test_pod_config_validation() {
        let valid = PODAttentionConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = PODAttentionConfig {
            prefill_sm_ratio: 0.8,
            decode_sm_ratio: 0.5, // Sum > 1.0
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid2 = PODAttentionConfig {
            min_sm_per_task: 100,
            total_sms: 108, // Not enough for 2 tasks
            ..Default::default()
        };
        assert!(invalid2.validate().is_err());
    }

    #[test]
    fn test_pod_config_sm_allocation() {
        let config = PODAttentionConfig::default();

        let prefill = config.prefill_sms();
        let decode = config.decode_sms();

        assert!(prefill >= config.min_sm_per_task);
        assert!(decode >= config.min_sm_per_task);
        assert!(prefill + decode <= config.total_sms);
    }

    #[test]
    fn test_pod_config_dynamic_adjustment() {
        let mut config = PODAttentionConfig::default();
        let initial_prefill = config.prefill_sm_ratio;

        // Heavy prefill load
        config.adjust_for_load(100, 10);
        assert!(config.prefill_sm_ratio != initial_prefill);

        // Heavy decode load
        config.adjust_for_load(10, 100);
    }

    #[test]
    fn test_scheduler_basic() {
        let chunk_config = ChunkConfig::default();
        let pod_config = PODAttentionConfig::default();
        let mut scheduler = ChunkedPrefillScheduler::new(chunk_config, pod_config).unwrap();

        // Submit prefill
        let tokens: Vec<TokenId> = (0..5000).collect();
        let request = PrefillRequest::new(1, tokens);
        scheduler.submit_prefill(request);

        // Submit decode
        let decode = DecodeRequest::new(2, 100, 500);
        scheduler.submit_decode(decode);

        assert!(scheduler.has_work());

        // Schedule batch
        let batch = scheduler.schedule_batch();
        assert!(!batch.is_empty());
        assert!(!batch.prefill_chunks.is_empty());
        assert!(!batch.decode_requests.is_empty());
    }

    #[test]
    fn test_scheduler_chunking() {
        let chunk_config = ChunkConfig {
            chunk_size: 1000,
            max_chunks_per_batch: 2,
            dynamic_chunk_size: false,
            ..Default::default()
        };
        let pod_config = PODAttentionConfig::default();
        let mut scheduler = ChunkedPrefillScheduler::new(chunk_config, pod_config).unwrap();

        // Submit large request
        let tokens: Vec<TokenId> = (0..5000).collect();
        scheduler.submit_prefill(PrefillRequest::new(1, tokens));

        // First batch: 2 chunks
        let batch1 = scheduler.schedule_batch();
        assert_eq!(batch1.prefill_chunks.len(), 2);
        assert_eq!(batch1.prefill_chunks[0].start_pos, 0);
        assert_eq!(batch1.prefill_chunks[0].end_pos, 1000);
        assert_eq!(batch1.prefill_chunks[1].start_pos, 1000);
        assert_eq!(batch1.prefill_chunks[1].end_pos, 2000);

        // Second batch: 2 more chunks
        let batch2 = scheduler.schedule_batch();
        assert_eq!(batch2.prefill_chunks.len(), 2);

        // Third batch: 1 remaining chunk
        let batch3 = scheduler.schedule_batch();
        assert_eq!(batch3.prefill_chunks.len(), 1);
        assert!(batch3.prefill_chunks[0].is_last);
    }

    #[test]
    fn test_scheduler_priority() {
        let chunk_config = ChunkConfig::default();
        let pod_config = PODAttentionConfig::default();
        let mut scheduler = ChunkedPrefillScheduler::new(chunk_config, pod_config).unwrap();

        // Submit low priority first
        let low_priority = PrefillRequest::new(1, vec![1, 2, 3]).with_priority(1);
        scheduler.submit_prefill(low_priority);

        // Submit high priority second
        let high_priority = PrefillRequest::new(2, vec![4, 5, 6]).with_priority(10);
        scheduler.submit_prefill(high_priority);

        // High priority should be scheduled first
        let batch = scheduler.schedule_batch();
        assert_eq!(batch.prefill_chunks[0].request_id, 2);
    }

    #[test]
    fn test_scheduler_execute_batch() {
        let chunk_config = ChunkConfig::default();
        let pod_config = PODAttentionConfig::default();
        let mut scheduler = ChunkedPrefillScheduler::new(chunk_config, pod_config).unwrap();

        scheduler.submit_prefill(PrefillRequest::new(1, vec![1, 2, 3]));
        scheduler.submit_decode(DecodeRequest::new(2, 100, 500));

        let batch = scheduler.schedule_batch();
        let output = scheduler.execute_batch(&batch);

        assert_eq!(output.prefill_outputs.len(), 1);
        assert_eq!(output.decode_outputs.len(), 1);
    }

    #[test]
    fn test_scheduled_batch_metrics() {
        let batch = ScheduledBatch {
            prefill_chunks: vec![
                PrefillChunk {
                    request_id: 1,
                    tokens: vec![1, 2, 3, 4, 5],
                    start_pos: 0,
                    end_pos: 5,
                    chunk_idx: 0,
                    total_chunks: 1,
                    is_last: true,
                },
            ],
            decode_requests: vec![
                DecodeRequest::new(2, 100, 500),
                DecodeRequest::new(3, 101, 600),
            ],
            prefill_sms: 60,
            decode_sms: 40,
            batch_id: 1,
            scheduled_at: Instant::now(),
        };

        assert_eq!(batch.prefill_tokens(), 5);
        assert_eq!(batch.decode_tokens(), 2);
        assert_eq!(batch.total_tokens(), 7);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_chunk_iterator() {
        let tokens: Vec<TokenId> = (0..10).collect();

        let chunks: Vec<_> = chunk_tokens(&tokens, 3).collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], &[0, 1, 2]);
        assert_eq!(chunks[1], &[3, 4, 5]);
        assert_eq!(chunks[2], &[6, 7, 8]);
        assert_eq!(chunks[3], &[9]);
    }

    #[test]
    fn test_scheduler_stats() {
        let chunk_config = ChunkConfig::default();
        let pod_config = PODAttentionConfig::default();
        let mut scheduler = ChunkedPrefillScheduler::new(chunk_config, pod_config).unwrap();

        scheduler.submit_prefill(PrefillRequest::new(1, vec![1, 2, 3]));
        scheduler.submit_decode(DecodeRequest::new(2, 100, 500));

        let _batch = scheduler.schedule_batch();
        let stats = scheduler.stats();

        assert_eq!(stats.prefill_submitted, 1);
        assert_eq!(stats.decode_submitted, 1);
    }

    #[test]
    fn test_config_presets() {
        let short = ChunkConfig::for_short_prompts();
        let long = ChunkConfig::for_long_prompts();

        assert!(short.chunk_size < long.chunk_size);
        assert!(short.max_chunks_per_batch > long.max_chunks_per_batch);

        let prefill_heavy = PODAttentionConfig::prefill_heavy();
        let decode_heavy = PODAttentionConfig::decode_heavy();

        assert!(prefill_heavy.prefill_sm_ratio > decode_heavy.prefill_sm_ratio);
    }

    #[test]
    fn test_stats_calculations() {
        let stats = SchedulerStats {
            prefill_submitted: 100,
            prefill_completed: 90,
            decode_submitted: 200,
            decode_completed: 180,
            batches_scheduled: 50,
            tokens_processed: 10000,
        };

        assert!((stats.prefill_completion_rate() - 0.9).abs() < 0.001);
        assert!((stats.decode_completion_rate() - 0.9).abs() < 0.001);
        assert!((stats.avg_tokens_per_batch() - 200.0).abs() < 0.001);

        let improvement = stats.estimated_throughput_improvement();
        assert!(improvement >= 1.0);
        assert!(improvement <= 1.22);
    }
}
