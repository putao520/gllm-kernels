//! Swap Manager for GPU <-> CPU page migration.
//!
//! Provides LRU eviction while honoring Warm/Protected pages to reduce thrashing.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::backend_trait::{BackendError, BackendResult};
use crate::kernel_types::{PageId, PageMetadata, PageState, SwapConfig};

const DEFAULT_WARMUP_MS: u64 = 100;
const DEFAULT_MIN_WARM_ACCESS: usize = 2;
const DEFAULT_PROTECTED_ACCESS: usize = 4;
const DEFAULT_WORKING_SET_WINDOW_MS: u64 = 1_000;

/// CPU-side swap buffer storing evicted pages.
#[derive(Debug)]
pub struct CpuSwapBuffer {
    /// CPU-side page backups
    host_pages: Vec<Vec<f32>>,
    /// Available CPU page slots
    free_list: Vec<usize>,
    /// Page index -> CPU slot mapping
    page_to_slot: HashMap<usize, usize>,
}

impl CpuSwapBuffer {
    /// Create a new CPU swap buffer with capacity in MB.
    pub fn new(capacity_mb: usize, page_size_bytes: usize) -> Self {
        let capacity_bytes = capacity_mb.saturating_mul(1024 * 1024);
        let page_bytes = page_size_bytes.max(1);
        let capacity_pages = capacity_bytes / page_bytes;
        Self {
            host_pages: Vec::with_capacity(capacity_pages),
            free_list: (0..capacity_pages).collect(),
            page_to_slot: HashMap::new(),
        }
    }

    /// Allocate a CPU page slot and store data.
    pub fn allocate_slot(&mut self, page_index: usize, data: Vec<f32>) -> BackendResult<()> {
        if let Some(&existing_slot) = self.page_to_slot.get(&page_index) {
            if existing_slot < self.host_pages.len() {
                self.host_pages[existing_slot] = data;
                return Ok(());
            }
        }

        let slot = self
            .free_list
            .pop()
            .ok_or_else(|| BackendError::InvalidConfig("CPU swap buffer exhausted".into()))?;

        if slot >= self.host_pages.len() {
            self.host_pages.resize(slot + 1, Vec::new());
        }
        self.host_pages[slot] = data;
        self.page_to_slot.insert(page_index, slot);
        Ok(())
    }

    /// Free a CPU page slot and return stored data (if any).
    pub fn free_slot(&mut self, page_index: usize) -> Option<Vec<f32>> {
        if let Some(slot) = self.page_to_slot.remove(&page_index) {
            self.free_list.push(slot);
            if slot < self.host_pages.len() {
                let data = std::mem::take(&mut self.host_pages[slot]);
                return Some(data);
            }
        }
        None
    }

    /// Get page data.
    pub fn get(&self, page_index: usize) -> Option<&[f32]> {
        self.page_to_slot
            .get(&page_index)
            .and_then(|&slot| self.host_pages.get(slot))
            .map(|v| v.as_slice())
    }

    /// Get mutable page data.
    pub fn get_mut(&mut self, page_index: usize) -> Option<&mut Vec<f32>> {
        if let Some(&slot) = self.page_to_slot.get(&page_index) {
            self.host_pages.get_mut(slot)
        } else {
            None
        }
    }

    /// Check if a page exists in the swap buffer.
    pub fn contains(&self, page_index: usize) -> bool {
        self.page_to_slot.contains_key(&page_index)
    }

    /// Number of used slots.
    pub fn used_slots(&self) -> usize {
        self.page_to_slot.len()
    }
}

/// Swap Manager - manages GPU <-> CPU page migration.
#[derive(Debug)]
pub struct SwapManager {
    /// Page metadata
    metadata: Vec<PageMetadata>,
    /// CPU swap buffer
    cpu_buffer: CpuSwapBuffer,
    /// Swap config
    config: SwapConfig,
    /// Warm-up duration
    warmup_duration: Duration,
    /// Warm stage minimal accesses
    min_warm_access: usize,
    /// Access threshold to promote to Protected
    protected_access_threshold: usize,
    /// Working set window (protection expiry)
    working_set_window: Duration,
}

impl SwapManager {
    /// Create a new Swap Manager.
    pub fn new(total_pages: usize, config: SwapConfig) -> Self {
        let now = Instant::now();
        let metadata = (0..total_pages)
            .map(|idx| PageMetadata {
                page_id: idx,
                sequence_id: None,
                state: PageState::Standby,
                recency: 0,
                is_lir: false,
                swap_in_time: None,
                warm_until: None,
                access_count: 0,
                last_access: now,
            })
            .collect();

        Self {
            metadata,
            cpu_buffer: CpuSwapBuffer::new(config.cpu_reserve_mb, config.page_size_bytes),
            config,
            warmup_duration: Duration::from_millis(DEFAULT_WARMUP_MS),
            min_warm_access: DEFAULT_MIN_WARM_ACCESS,
            protected_access_threshold: DEFAULT_PROTECTED_ACCESS,
            working_set_window: Duration::from_millis(DEFAULT_WORKING_SET_WINDOW_MS),
        }
    }

    fn meta(&self, page_index: PageId) -> Option<&PageMetadata> {
        self.metadata.get(page_index)
    }

    fn meta_mut(&mut self, page_index: PageId) -> Option<&mut PageMetadata> {
        self.metadata.get_mut(page_index)
    }

    /// Page size in bytes.
    pub fn page_size_bytes(&self) -> usize {
        self.config.page_size_bytes.max(1)
    }

    /// Get page state.
    pub fn get_page_state(&self, page_index: usize) -> Option<PageState> {
        self.meta(page_index).map(|m| m.state)
    }

    /// Set page state.
    pub fn set_page_state(&mut self, page_index: usize, state: PageState) -> BackendResult<()> {
        let Some(meta) = self.meta_mut(page_index) else {
            return Err(BackendError::InvalidHandle(
                "page index out of range".into(),
            ));
        };
        meta.state = state;
        if state == PageState::Swapped {
            meta.swap_in_time = None;
            meta.warm_until = None;
        }
        Ok(())
    }

    /// Mark a page as accessed (LRU update).
    pub fn mark_accessed(&mut self, page_index: usize) -> BackendResult<()> {
        if page_index >= self.metadata.len() {
            return Err(BackendError::InvalidHandle(
                "page index out of range".into(),
            ));
        }

        let now = Instant::now();
        let min_warm_access = self.min_warm_access;
        let protected_threshold = self.protected_access_threshold;
        let warmup_duration = self.warmup_duration;

        {
            let meta = self
                .meta_mut(page_index)
                .expect("metadata length checked above");

            if meta.state == PageState::Swapped {
                return Err(BackendError::InvalidHandle(
                    "page is swapped out; cannot mark accessed".into(),
                ));
            }

            let irr = now
                .saturating_duration_since(meta.last_access)
                .as_millis()
                .try_into()
                .unwrap_or(usize::MAX);
            meta.recency = irr;
            meta.access_count = meta.access_count.saturating_add(1);
            meta.last_access = now;

            if meta.state == PageState::Standby {
                meta.state = PageState::Active;
            }

            if meta.state == PageState::Warm
                && !Self::is_in_warm_period_meta(meta, now, warmup_duration, min_warm_access)
            {
                meta.state = PageState::Active;
                meta.warm_until = None;
            }

            if meta.state == PageState::Active && meta.access_count >= protected_threshold {
                meta.state = PageState::Protected;
                meta.warm_until = None;
            }
        }

        Ok(())
    }

    /// Calculate memory pressure (0.0 - 1.0).
    pub fn calculate_memory_pressure(&self, current_usage: usize, total_memory: usize) -> f32 {
        if total_memory == 0 {
            return 1.0;
        }
        let pressure = current_usage as f32 / total_memory as f32;
        pressure.clamp(0.0, 1.0)
    }

    /// Determine whether swap-out should be triggered.
    pub fn needs_swap_out(
        &self,
        current_usage: usize,
        total_memory: usize,
        required_bytes: usize,
    ) -> bool {
        let pressure = self.calculate_memory_pressure(current_usage, total_memory);
        let threshold = self.config.swap_threshold;
        pressure >= threshold || current_usage.saturating_add(required_bytes) > total_memory
    }

    /// Check memory pressure and select victim pages for swap-out.
    pub fn check_and_auto_swap(
        &mut self,
        current_usage: usize,
        total_memory: usize,
        bytes_to_free: usize,
    ) -> BackendResult<Vec<usize>> {
        if !self.needs_swap_out(current_usage, total_memory, bytes_to_free) {
            return Ok(Vec::new());
        }

        let page_size = self.page_size_bytes();
        let pages_needed = (bytes_to_free + page_size - 1) / page_size;
        let victims = self.select_victim_pages(pages_needed.max(1));

        if victims.is_empty() {
            return Err(BackendError::OutOfMemory(
                "No swappable pages available (all pages are Warm/Protected)".into(),
            ));
        }

        Ok(victims)
    }

    fn is_in_warm_period_meta(
        meta: &PageMetadata,
        now: Instant,
        warmup_duration: Duration,
        min_warm_access: usize,
    ) -> bool {
        if meta.state != PageState::Warm {
            return false;
        }
        if meta.access_count >= min_warm_access {
            return false;
        }
        let warm_until = meta
            .warm_until
            .or_else(|| meta.swap_in_time.map(|t| t + warmup_duration));
        warm_until.map(|end| now < end).unwrap_or(false)
    }

    fn is_protection_expired(meta: &PageMetadata, now: Instant, window: Duration) -> bool {
        if meta.state != PageState::Protected {
            return false;
        }
        now.saturating_duration_since(meta.last_access) >= window
    }

    /// Check whether a page is in Warm/Protected protection period.
    pub fn is_in_protected_period(&self, page_index: usize) -> bool {
        let now = Instant::now();
        self.meta(page_index)
            .map(|m| {
                Self::is_in_warm_period_meta(m, now, self.warmup_duration, self.min_warm_access)
                    || m.state == PageState::Protected
            })
            .unwrap_or(false)
    }

    /// Select pages to evict (LRU), skipping Warm/Protected/Swapped.
    pub fn select_victim_pages(&mut self, count: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let now = Instant::now();
        let warmup_duration = self.warmup_duration;
        let min_warm_access = self.min_warm_access;
        let working_set_window = self.working_set_window;

        let mut candidates: Vec<(usize, Instant)> = self
            .metadata
            .iter_mut()
            .enumerate()
            .filter_map(|(page_idx, meta)| {
                if meta.state == PageState::Warm
                    && !Self::is_in_warm_period_meta(meta, now, warmup_duration, min_warm_access)
                {
                    meta.state = PageState::Active;
                    meta.warm_until = None;
                }
                if Self::is_protection_expired(meta, now, working_set_window) {
                    meta.state = PageState::Standby;
                }

                match meta.state {
                    PageState::Active | PageState::Standby => Some((page_idx, meta.last_access)),
                    _ => None,
                }
            })
            .collect();

        candidates.sort_by_key(|(_, last_access)| *last_access);
        candidates
            .into_iter()
            .take(count)
            .map(|(page_idx, _)| page_idx)
            .collect()
    }

    /// Prepare swap-out: returns page indices that are eligible to swap.
    /// Actual GPU -> CPU copy is performed by the caller.
    pub fn prepare_swap_out(&mut self, page_indices: &[usize]) -> BackendResult<Vec<usize>> {
        let mut prepared = Vec::new();
        let now = Instant::now();
        let min_warm_access = self.min_warm_access;
        let warmup_duration = self.warmup_duration;
        for &page_idx in page_indices {
            let Some(meta) = self.meta(page_idx) else {
                continue;
            };
            if matches!(meta.state, PageState::Swapped) {
                continue;
            }
            if Self::is_in_warm_period_meta(meta, now, warmup_duration, min_warm_access)
                || meta.state == PageState::Protected
            {
                continue;
            }
            prepared.push(page_idx);
        }
        Ok(prepared)
    }

    /// Complete swap-out: mark page as swapped out and store data.
    pub fn complete_swap_out(&mut self, page_idx: usize, data: Vec<f32>) -> BackendResult<()> {
        self.cpu_buffer.allocate_slot(page_idx, data)?;
        if let Some(meta) = self.meta_mut(page_idx) {
            meta.state = PageState::Swapped;
            meta.warm_until = None;
            meta.swap_in_time = None;
            meta.access_count = 0;
            meta.recency = 0;
        }
        Ok(())
    }

    /// Prepare swap-in: return CPU data if available.
    pub fn prepare_swap_in(&self, page_idx: usize) -> Option<&[f32]> {
        if self.is_swapped_out(page_idx) {
            self.cpu_buffer.get(page_idx)
        } else {
            None
        }
    }

    /// Complete swap-in: mark page as warm and reset counters.
    pub fn complete_swap_in(&mut self, page_idx: usize) -> BackendResult<()> {
        self.cpu_buffer.free_slot(page_idx);
        let now = Instant::now();
        let warm_until = now + self.warmup_duration;

        let Some(meta) = self.meta_mut(page_idx) else {
            return Err(BackendError::InvalidHandle(
                "page index out of range".into(),
            ));
        };

        meta.state = PageState::Warm;
        meta.swap_in_time = Some(now);
        meta.warm_until = Some(warm_until);
        meta.access_count = 0;
        meta.last_access = now;
        meta.recency = 0;
        Ok(())
    }

    /// Get CPU data for a swapped out page.
    pub fn get_swapped_data(&self, page_idx: usize) -> Option<&[f32]> {
        self.cpu_buffer.get(page_idx)
    }

    /// Remove swapped data from CPU buffer.
    pub fn remove_swapped_data(&mut self, page_idx: usize) -> Option<Vec<f32>> {
        if self.is_swapped_out(page_idx) {
            self.cpu_buffer.free_slot(page_idx)
        } else {
            None
        }
    }

    /// Check if a page is swapped out.
    pub fn is_swapped_out(&self, page_index: usize) -> bool {
        self.get_page_state(page_index) == Some(PageState::Swapped)
    }

    /// Get all swapped out pages.
    pub fn swapped_out_pages(&self) -> Vec<usize> {
        self.metadata
            .iter()
            .enumerate()
            .filter_map(|(idx, meta)| (meta.state == PageState::Swapped).then_some(idx))
            .collect()
    }

    /// Mark a page as Standby (unused but still on GPU).
    pub fn mark_standby(&mut self, page_index: usize) -> BackendResult<()> {
        let Some(meta) = self.meta_mut(page_index) else {
            return Err(BackendError::InvalidHandle(
                "page index out of range".into(),
            ));
        };
        if meta.state == PageState::Active {
            meta.state = PageState::Standby;
        }
        Ok(())
    }

    /// Memory usage stats.
    pub fn stats(&self) -> SwapStats {
        let mut active = 0;
        let mut standby = 0;
        let mut warm = 0;
        let mut protected = 0;
        let mut swapped = 0;

        for meta in &self.metadata {
            match meta.state {
                PageState::Active => active += 1,
                PageState::Standby => standby += 1,
                PageState::Warm => warm += 1,
                PageState::Protected => protected += 1,
                PageState::Swapped => swapped += 1,
            }
        }

        SwapStats {
            total_pages: self.metadata.len(),
            active_pages: active,
            standby_pages: standby,
            warm_pages: warm,
            protected_pages: protected,
            swapped_pages: swapped,
            cpu_buffer_used: self.cpu_buffer.used_slots(),
        }
    }

    /// Cleanup all swapped pages.
    pub fn cleanup(&mut self) {
        for page_idx in self.swapped_out_pages() {
            self.cpu_buffer.free_slot(page_idx);
            if let Some(meta) = self.meta_mut(page_idx) {
                meta.state = PageState::Standby;
                meta.swap_in_time = None;
                meta.warm_until = None;
                meta.access_count = 0;
                meta.recency = 0;
            }
        }
    }
}

/// Swap stats.
#[derive(Debug, Clone)]
pub struct SwapStats {
    pub total_pages: usize,
    pub active_pages: usize,
    pub standby_pages: usize,
    pub warm_pages: usize,
    pub protected_pages: usize,
    pub swapped_pages: usize,
    pub cpu_buffer_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_swap_buffer() {
        let mut buffer = CpuSwapBuffer::new(1, 4096); // 1 MB
        let data = vec![1.0f32; 256];

        buffer.allocate_slot(0, data.clone()).unwrap();
        assert!(buffer.contains(0));
        assert_eq!(buffer.get(0), Some(data.as_slice()));

        let recovered = buffer.free_slot(0);
        assert!(!buffer.contains(0));
        assert_eq!(recovered, Some(data));
    }

    #[test]
    fn test_selection_respects_page_states() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(6, config);

        manager.set_page_state(0, PageState::Protected).unwrap();
        manager.complete_swap_in(1).unwrap(); // Warm
        manager.set_page_state(2, PageState::Swapped).unwrap();

        let victims = manager.select_victim_pages(2);
        assert_eq!(victims.len(), 2);
        assert!(!victims.contains(&0));
        assert!(!victims.contains(&1));
        assert!(!victims.contains(&2));
    }

    #[test]
    fn test_warm_and_protected_not_evicted() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(4, config);

        for i in 0..4 {
            manager.mark_accessed(i).unwrap();
        }

        manager.complete_swap_in(0).unwrap();
        for _ in 0..5 {
            manager.mark_accessed(1).unwrap();
        }

        let victims = manager.select_victim_pages(2);
        assert!(!victims.contains(&0));
        assert!(!victims.contains(&1));
    }

    #[test]
    fn test_page_state_transitions() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(5, config);

        assert_eq!(manager.get_page_state(0), Some(PageState::Standby));

        manager.mark_standby(0).unwrap();
        assert_eq!(manager.get_page_state(0), Some(PageState::Standby));

        manager.mark_accessed(0).unwrap();
        assert_eq!(manager.get_page_state(0), Some(PageState::Active));
    }

    #[test]
    fn test_swap_stats() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(10, config);

        for i in 0..3 {
            manager.mark_standby(i).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.total_pages, 10);
        assert_eq!(stats.active_pages, 0);
        assert_eq!(stats.standby_pages, 10);
        assert_eq!(stats.warm_pages, 0);
        assert_eq!(stats.protected_pages, 0);
        assert_eq!(stats.swapped_pages, 0);
    }

    #[test]
    fn test_calculate_memory_pressure() {
        let manager = SwapManager::new(1, SwapConfig::default());
        assert_eq!(manager.calculate_memory_pressure(0, 100), 0.0);
        assert!((manager.calculate_memory_pressure(50, 100) - 0.5).abs() < 1e-6);
        assert_eq!(manager.calculate_memory_pressure(1, 0), 1.0);
    }

    #[test]
    fn test_needs_swap_out() {
        let manager = SwapManager::new(1, SwapConfig::default());
        let total = 1_000usize;
        assert!(!manager.needs_swap_out(500, total, 100));
        assert!(manager.needs_swap_out(900, total, 0));
        assert!(manager.needs_swap_out(800, total, 300));
    }

    #[test]
    fn test_check_and_auto_swap_marks_victims() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(4, config);

        for _ in 0..config.lru_granularity + 1 {
            manager.mark_accessed(1).unwrap();
        }
        manager.complete_swap_in(0).unwrap();

        let victims = manager
            .check_and_auto_swap(900, 1000, 2048)
            .expect("should select victims");
        assert_eq!(victims.len(), 1);
        assert!(!victims.contains(&0));
        assert!(!victims.contains(&1));
    }

    #[test]
    fn test_check_and_auto_swap_no_victim_error() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(2, config);

        manager.set_page_state(0, PageState::Protected).unwrap();
        manager.set_page_state(1, PageState::Protected).unwrap();

        let err = manager
            .check_and_auto_swap(900, 1000, 2048)
            .expect_err("should fail without swappable pages");
        assert!(matches!(err, BackendError::OutOfMemory(_)));
    }
}
