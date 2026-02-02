//! Swap Manager for GPU <-> CPU page migration.
//!
//! Implements LRU-based page swapping to extend effective GPU memory capacity.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::backend_trait::{BackendError, BackendResult};
use crate::kernel_types::{PageId, PageMetadata, PageState, SwapConfig};

/// CPU 端 Swap 缓冲区
#[derive(Debug)]
pub struct CpuSwapBuffer {
    /// CPU 端页面备份
    host_pages: Vec<Vec<f32>>,
    /// 可用的 CPU 页槽索引
    free_list: Vec<usize>,
    /// 页索引 -> CPU 槽索引的映射
    page_to_slot: HashMap<usize, usize>,
}

impl CpuSwapBuffer {
    /// 创建新的 CPU swap buffer
    pub fn new(capacity_mb: usize) -> Self {
        let capacity_pages = capacity_mb * 1024 * 1024 / 4; // f32 = 4 bytes
        Self {
            host_pages: Vec::with_capacity(capacity_pages),
            free_list: (0..capacity_pages).collect(),
            page_to_slot: HashMap::new(),
        }
    }

    /// 分配一个 CPU 页槽
    pub fn allocate_slot(&mut self, page_index: usize, data: Vec<f32>) -> BackendResult<()> {
        if let Some(&existing_slot) = self.page_to_slot.get(&page_index) {
            // 页面已存在，覆盖数据
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

    /// 释放一个 CPU 页槽
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

    /// 获取页面数据
    pub fn get(&self, page_index: usize) -> Option<&[f32]> {
        self.page_to_slot
            .get(&page_index)
            .and_then(|&slot| self.host_pages.get(slot))
            .map(|v| v.as_slice())
    }

    /// 获取可变页面数据
    pub fn get_mut(&mut self, page_index: usize) -> Option<&mut Vec<f32>> {
        if let Some(&slot) = self.page_to_slot.get(&page_index) {
            self.host_pages.get_mut(slot)
        } else {
            None
        }
    }

    /// 检查页面是否存在
    pub fn contains(&self, page_index: usize) -> bool {
        self.page_to_slot.contains_key(&page_index)
    }

    /// 获取已使用槽位数
    pub fn used_slots(&self) -> usize {
        self.page_to_slot.len()
    }
}

/// Swap Manager - 管理 GPU <-> CPU 页面迁移
#[derive(Debug)]
pub struct SwapManager {
    /// 页面元数据
    metadata: Vec<PageMetadata>,
    /// LRU 队列
    lru_queue: VecDeque<usize>,
    /// CPU swap buffer
    cpu_buffer: CpuSwapBuffer,
    /// 全局 IRR 计数
    current_recency: usize,
    /// Swap 配置
    #[allow(dead_code)]
    config: SwapConfig,
    /// Warm 保护时长
    warmup_duration: Duration,
    /// Warm 阶段最少访问次数
    min_warm_access: usize,
    /// 晋升为 Protected 所需访问次数
    protected_access_threshold: usize,
}

impl SwapManager {
    /// 创建新的 Swap Manager
    pub fn new(total_pages: usize, config: SwapConfig) -> Self {
        let now = Instant::now();
        let metadata = (0..total_pages)
            .map(|idx| PageMetadata {
                page_id: idx,
                sequence_id: None,
                state: PageState::Active,
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
            lru_queue: (0..total_pages).collect(),
            cpu_buffer: CpuSwapBuffer::new(config.cpu_reserve_mb),
            current_recency: 0,
            config,
            warmup_duration: Duration::from_millis(100),
            min_warm_access: 2,
            protected_access_threshold: 4,
        }
    }

    fn meta(&self, page_index: PageId) -> Option<&PageMetadata> {
        self.metadata.get(page_index)
    }

    fn meta_mut(&mut self, page_index: PageId) -> Option<&mut PageMetadata> {
        self.metadata.get_mut(page_index)
    }

    fn touch_lru(&mut self, page_index: PageId) {
        self.lru_queue.retain(|&idx| idx != page_index);
        self.lru_queue.push_back(page_index);
    }

    fn remove_from_lru(&mut self, page_index: PageId) {
        self.lru_queue.retain(|&idx| idx != page_index);
    }

    /// 获取页面状态
    pub fn get_page_state(&self, page_index: usize) -> Option<PageState> {
        self.meta(page_index).map(|m| m.state)
    }

    /// 设置页面状态
    pub fn set_page_state(&mut self, page_index: usize, state: PageState) -> BackendResult<()> {
        let Some(meta) = self.meta_mut(page_index) else {
            return Err(BackendError::InvalidHandle(
                "page index out of range".into(),
            ));
        };
        meta.state = state;
        Ok(())
    }

    /// 标记页面访问 (更新 LRU)
    pub fn mark_accessed(&mut self, page_index: usize) -> BackendResult<()> {
        if page_index >= self.metadata.len() {
            return Err(BackendError::InvalidHandle(
                "page index out of range".into(),
            ));
        }

        let now = Instant::now();
        self.current_recency = self.current_recency.wrapping_add(1);
        let recency = self.current_recency;
        let min_warm_access = self.min_warm_access;
        let protected_threshold = self.protected_access_threshold;

        {
            let meta = self
                .meta_mut(page_index)
                .expect("metadata length checked above");

            if meta.state == PageState::Swapped {
                return Err(BackendError::InvalidHandle(
                    "page is swapped out; cannot mark accessed".into(),
                ));
            }

            meta.recency = recency;
            meta.access_count = meta.access_count.saturating_add(1);
            meta.last_access = now;

            if meta.state == PageState::Standby {
                meta.state = PageState::Active;
            }

            if meta.state == PageState::Warm
                && !Self::is_in_protected_period_meta(meta, now, min_warm_access)
            {
                meta.state = PageState::Active;
                meta.warm_until = None;
            }

            if meta.state == PageState::Active && meta.access_count >= protected_threshold {
                meta.state = PageState::Protected;
                meta.warm_until = None;
            }
        }
        self.touch_lru(page_index);

        Ok(())
    }

    fn is_in_protected_period_meta(
        meta: &PageMetadata,
        now: Instant,
        min_warm_access: usize,
    ) -> bool {
        match meta.state {
            PageState::Warm => {
                let in_time_window = meta.warm_until.map(|until| until > now).unwrap_or(false);
                let needs_more_access = meta.access_count < min_warm_access;
                in_time_window && needs_more_access
            }
            PageState::Protected => true,
            _ => false,
        }
    }

    /// 检查页面是否处于 Warm/Protected 保护期
    pub fn is_in_protected_period(&self, page_index: usize) -> bool {
        let now = Instant::now();
        self.meta(page_index)
            .map(|m| Self::is_in_protected_period_meta(m, now, self.min_warm_access))
            .unwrap_or(false)
    }

    /// 选择要换出的页面 (基于 LRU)
    pub fn select_victim_pages(&mut self, count: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        let now = Instant::now();
        let min_warm_access = self.min_warm_access;

        let queue_snapshot: Vec<_> = self.lru_queue.iter().copied().collect();
        for page_idx in queue_snapshot {
            if victims.len() >= count {
                break;
            }
            if let Some(meta) = self.meta_mut(page_idx) {
                if meta.state == PageState::Warm
                    && !Self::is_in_protected_period_meta(meta, now, min_warm_access)
                {
                    meta.state = PageState::Active;
                    meta.warm_until = None;
                }
                match meta.state {
                    PageState::Active | PageState::Standby => {
                        victims.push(page_idx);
                    }
                    _ => {}
                }
            }
        }

        victims
    }

    /// 准备 swap-out: 返回需要拷贝的页面索引
    /// 实际的 GPU -> CPU 拷贝由调用者完成
    pub fn prepare_swap_out(&mut self, page_indices: &[usize]) -> BackendResult<Vec<usize>> {
        let mut prepared = Vec::new();
        let now = Instant::now();
        let min_warm_access = self.min_warm_access;
        for &page_idx in page_indices {
            let Some(meta) = self.meta(page_idx) else {
                continue;
            };
            if matches!(meta.state, PageState::Swapped) {
                continue;
            }
            if Self::is_in_protected_period_meta(meta, now, min_warm_access) {
                continue;
            }
            prepared.push(page_idx);
        }
        Ok(prepared)
    }

    /// 完成 swap-out: 标记页面为已换出
    pub fn complete_swap_out(&mut self, page_idx: usize, data: Vec<f32>) -> BackendResult<()> {
        self.cpu_buffer.allocate_slot(page_idx, data)?;
        if let Some(meta) = self.meta_mut(page_idx) {
            meta.state = PageState::Swapped;
            meta.warm_until = None;
            meta.swap_in_time = None;
        }
        self.remove_from_lru(page_idx);
        Ok(())
    }

    /// 准备 swap-in: 返回 CPU 数据
    pub fn prepare_swap_in(&self, page_idx: usize) -> Option<&[f32]> {
        if self.is_swapped_out(page_idx) {
            self.cpu_buffer.get(page_idx)
        } else {
            None
        }
    }

    /// 完成 swap-in: 标记页面为已换入
    pub fn complete_swap_in(&mut self, page_idx: usize) -> BackendResult<()> {
        self.cpu_buffer.free_slot(page_idx);
        let now = Instant::now();
        let warm_until = now + self.warmup_duration;
        self.current_recency = self.current_recency.wrapping_add(1);
        let recency = self.current_recency;

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
        meta.recency = recency;

        self.touch_lru(page_idx);
        Ok(())
    }

    /// 获取换出页面的 CPU 数据（用于 swap-in）
    pub fn get_swapped_data(&self, page_idx: usize) -> Option<&[f32]> {
        self.cpu_buffer.get(page_idx)
    }

    /// 删除换出页面的 CPU 数据
    pub fn remove_swapped_data(&mut self, page_idx: usize) -> Option<Vec<f32>> {
        if self.is_swapped_out(page_idx) {
            self.cpu_buffer.free_slot(page_idx)
        } else {
            None
        }
    }

    /// 检查页面是否已换出
    pub fn is_swapped_out(&self, page_index: usize) -> bool {
        self.get_page_state(page_index) == Some(PageState::Swapped)
    }

    /// 获取所有已换出的页面
    pub fn swapped_out_pages(&self) -> Vec<usize> {
        self.metadata
            .iter()
            .enumerate()
            .filter_map(|(idx, meta)| (meta.state == PageState::Swapped).then_some(idx))
            .collect()
    }

    /// 标记页面为 Standby (未使用但仍在 GPU)
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

    /// 获取内存使用统计
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

    /// 清理所有已换出的页面
    pub fn cleanup(&mut self) {
        for page_idx in self.swapped_out_pages() {
            self.cpu_buffer.free_slot(page_idx);
            if let Some(meta) = self.meta_mut(page_idx) {
                meta.state = PageState::Active;
                meta.swap_in_time = None;
                meta.warm_until = None;
                meta.access_count = 0;
            }
        }
    }
}

/// Swap 统计信息
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
        let mut buffer = CpuSwapBuffer::new(1); // 1 MB
        let data = vec![1.0f32; 256];

        // 分配
        buffer.allocate_slot(0, data.clone()).unwrap();
        assert!(buffer.contains(0));
        assert_eq!(buffer.get(0), Some(data.as_slice()));

        // 释放
        let recovered = buffer.free_slot(0);
        assert!(!buffer.contains(0));
        assert_eq!(recovered, Some(data));
    }

    #[test]
    fn test_lru_selection() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(10, config);

        // 标记访问顺序: 0, 1, 2, 3, 4
        // LRU 队列状态变为: [5,6,7,8,9,0,1,2,3,4]
        // (最久未使用在前端，最近使用在末端)
        for i in 0..5 {
            manager.mark_accessed(i).unwrap();
        }

        // 选择 2 个 victim，应该是 5 和 6 (从未访问过的页面)
        let victims = manager.select_victim_pages(2);
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0], 5);
        assert_eq!(victims[1], 6);
    }

    #[test]
    fn test_warm_and_protected_not_evicted() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(4, config);

        // mark pages to order LRU as [0,1,2,3]
        for i in 0..4 {
            manager.mark_accessed(i).unwrap();
        }

        // put page 0 into Warm (simulating swap-in)
        manager.complete_swap_in(0).unwrap();
        // mark page 1 heavily to become Protected
        for _ in 0..5 {
            manager.mark_accessed(1).unwrap();
        }

        // victims should skip 0 (Warm) and 1 (Protected)
        let victims = manager.select_victim_pages(2);
        assert!(!victims.contains(&0));
        assert!(!victims.contains(&1));
    }

    #[test]
    fn test_page_state_transitions() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(5, config);

        // 初始状态都是 Active
        assert_eq!(manager.get_page_state(0), Some(PageState::Active));

        // 标记为 Standby
        manager.mark_standby(0).unwrap();
        assert_eq!(manager.get_page_state(0), Some(PageState::Standby));

        // 访问后变回 Active
        manager.mark_accessed(0).unwrap();
        assert_eq!(manager.get_page_state(0), Some(PageState::Active));
    }

    #[test]
    fn test_swap_stats() {
        let config = SwapConfig::default();
        let mut manager = SwapManager::new(10, config);

        // 标记一些为 Standby
        for i in 0..3 {
            manager.mark_standby(i).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.total_pages, 10);
        assert_eq!(stats.active_pages, 7);
        assert_eq!(stats.standby_pages, 3);
        assert_eq!(stats.warm_pages, 0);
        assert_eq!(stats.protected_pages, 0);
        assert_eq!(stats.swapped_pages, 0);
    }
}
