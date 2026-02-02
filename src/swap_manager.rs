//! Swap Manager for GPU <-> CPU page migration.
//!
//! Implements LRU-based page swapping to extend effective GPU memory capacity.

use std::collections::{HashMap, VecDeque};

use crate::backend_trait::{BackendError, BackendResult};
use crate::kernel_types::{PageState, SwapConfig};

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

        let slot = self.free_list.pop()
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
        self.page_to_slot.get(&page_index)
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
    /// 页面状态
    page_states: Vec<PageState>,
    /// 访问时间戳
    access_times: Vec<u64>,
    /// LRU 队列
    lru_queue: VecDeque<usize>,
    /// CPU swap buffer
    cpu_buffer: CpuSwapBuffer,
    /// 全局时间戳
    current_time: u64,
    /// Swap 配置
    #[allow(dead_code)]
    config: SwapConfig,
}

impl SwapManager {
    /// 创建新的 Swap Manager
    pub fn new(total_pages: usize, config: SwapConfig) -> Self {
        Self {
            page_states: vec![PageState::Active; total_pages],
            access_times: vec![0; total_pages],
            lru_queue: (0..total_pages).collect(),
            cpu_buffer: CpuSwapBuffer::new(config.cpu_reserve_mb),
            current_time: 0,
            config,
        }
    }

    /// 获取页面状态
    pub fn get_page_state(&self, page_index: usize) -> Option<PageState> {
        self.page_states.get(page_index).copied()
    }

    /// 设置页面状态
    pub fn set_page_state(&mut self, page_index: usize, state: PageState) -> BackendResult<()> {
        if page_index >= self.page_states.len() {
            return Err(BackendError::InvalidHandle("page index out of range".into()));
        }
        self.page_states[page_index] = state;
        Ok(())
    }

    /// 标记页面访问 (更新 LRU)
    pub fn mark_accessed(&mut self, page_index: usize) -> BackendResult<()> {
        if page_index >= self.page_states.len() {
            return Err(BackendError::InvalidHandle("page index out of range".into()));
        }

        self.current_time = self.current_time.wrapping_add(1);
        self.access_times[page_index] = self.current_time;

        // 从 LRU 队列中移除
        self.lru_queue.retain(|&idx| idx != page_index);

        // 如果页面是 Standby，标记为 Active
        if self.page_states[page_index] == PageState::Standby {
            self.page_states[page_index] = PageState::Active;
        }

        // 添加到队列末尾 (最近使用)
        self.lru_queue.push_back(page_index);

        Ok(())
    }

    /// 选择要换出的页面 (基于 LRU)
    pub fn select_victim_pages(&self, count: usize) -> Vec<usize> {
        let mut victims = Vec::new();

        // 直接从 LRU 队列前端选择 (队列已按使用顺序排序)
        // LRU 队列最前端是最久未使用的页面
        for &page_idx in &self.lru_queue {
            if victims.len() >= count {
                break;
            }
            if self.page_states[page_idx] == PageState::Active
                || self.page_states[page_idx] == PageState::Standby
            {
                victims.push(page_idx);
            }
        }

        victims
    }

    /// 准备 swap-out: 返回需要拷贝的页面索引
    /// 实际的 GPU -> CPU 拷贝由调用者完成
    pub fn prepare_swap_out(&mut self, page_indices: &[usize]) -> BackendResult<Vec<usize>> {
        let mut prepared = Vec::new();
        for &page_idx in page_indices {
            if page_idx >= self.page_states.len() {
                continue;
            }
            if self.page_states[page_idx] == PageState::Swapped {
                continue; // 已换出
            }
            prepared.push(page_idx);
        }
        Ok(prepared)
    }

    /// 完成 swap-out: 标记页面为已换出
    pub fn complete_swap_out(&mut self, page_idx: usize, data: Vec<f32>) -> BackendResult<()> {
        self.cpu_buffer.allocate_slot(page_idx, data)?;
        self.page_states[page_idx] = PageState::Swapped;
        self.lru_queue.retain(|&idx| idx != page_idx);
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
        self.page_states[page_idx] = PageState::Active;
        self.mark_accessed(page_idx)?;
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
        self.page_states
            .iter()
            .enumerate()
            .filter_map(|(idx, &state)| {
                if state == PageState::Swapped {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// 标记页面为 Standby (未使用但仍在 GPU)
    pub fn mark_standby(&mut self, page_index: usize) -> BackendResult<()> {
        if page_index >= self.page_states.len() {
            return Err(BackendError::InvalidHandle("page index out of range".into()));
        }
        if self.page_states[page_index] == PageState::Active {
            self.page_states[page_index] = PageState::Standby;
        }
        Ok(())
    }

    /// 获取内存使用统计
    pub fn stats(&self) -> SwapStats {
        let active = self.page_states.iter()
            .filter(|&&s| s == PageState::Active)
            .count();
        let standby = self.page_states.iter()
            .filter(|&&s| s == PageState::Standby)
            .count();
        let swapped = self.page_states.iter()
            .filter(|&&s| s == PageState::Swapped)
            .count();

        SwapStats {
            total_pages: self.page_states.len(),
            active_pages: active,
            standby_pages: standby,
            swapped_pages: swapped,
            cpu_buffer_used: self.cpu_buffer.used_slots(),
        }
    }

    /// 清理所有已换出的页面
    pub fn cleanup(&mut self) {
        for page_idx in self.swapped_out_pages() {
            self.cpu_buffer.free_slot(page_idx);
            self.page_states[page_idx] = PageState::Active;
        }
    }
}

/// Swap 统计信息
#[derive(Debug, Clone)]
pub struct SwapStats {
    pub total_pages: usize,
    pub active_pages: usize,
    pub standby_pages: usize,
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
        assert_eq!(stats.swapped_pages, 0);
    }
}
