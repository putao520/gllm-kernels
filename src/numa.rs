//! NUMA topology detection and awareness for multi-socket servers.
//!
//! Reads `/sys/devices/system/node/` to discover:
//! - Number of NUMA nodes
//! - Which CPUs belong to each node
//! - Per-node memory capacity
//! - Inter-node distances (for locality-aware scheduling)
//!
//! On single-node systems (or when sysfs is unavailable), gracefully degrades
//! to a single-node topology covering all online CPUs.
//!
//! All results are cached in `OnceLock` -- detected once, zero overhead after.

use std::sync::OnceLock;

/// A single NUMA node's topology.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID (0-based, matches /sys/devices/system/node/nodeN)
    pub id: usize,
    /// Logical CPU IDs belonging to this node
    pub cpus: Vec<usize>,
    /// Total memory on this node in bytes (0 if unknown)
    pub mem_total: usize,
    /// Per-node L3 cache size in bytes (0 if unknown)
    pub l3_size: usize,
}

/// Complete NUMA topology of the system.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// All NUMA nodes, sorted by node ID
    pub nodes: Vec<NumaNode>,
    /// Distance matrix: distances[i][j] = distance from node i to node j
    /// Typically 10 for local, 20-21 for remote on dual-socket.
    /// Empty if distances unavailable.
    pub distances: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Number of NUMA nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Whether this is a multi-node (multi-socket) system.
    #[inline]
    pub fn is_multi_node(&self) -> bool {
        self.nodes.len() > 1
    }

    /// Total number of CPUs across all nodes.
    #[inline]
    pub fn total_cpus(&self) -> usize {
        self.nodes.iter().map(|n| n.cpus.len()).sum()
    }

    /// Find which NUMA node a given CPU belongs to.
    /// Returns None if CPU not found in any node.
    pub fn cpu_to_node(&self, cpu: usize) -> Option<usize> {
        for node in &self.nodes {
            if node.cpus.contains(&cpu) {
                return Some(node.id);
            }
        }
        None
    }

    /// Get per-node L3 cache size. On multi-socket systems, L3 is per-socket.
    /// Returns the L3 size for the given node, or the global L3 if per-node is unknown.
    pub fn node_l3_size(&self, node_id: usize) -> usize {
        if node_id < self.nodes.len() && self.nodes[node_id].l3_size > 0 {
            self.nodes[node_id].l3_size
        } else {
            crate::cache_params::l3_size()
        }
    }
}

static TOPOLOGY: OnceLock<NumaTopology> = OnceLock::new();

/// Get the cached NUMA topology. Detected once on first call.
pub fn topology() -> &'static NumaTopology {
    TOPOLOGY.get_or_init(detect_topology)
}

/// Returns true if the system has multiple NUMA nodes.
#[inline]
pub fn is_numa() -> bool {
    topology().is_multi_node()
}

/// Number of NUMA nodes.
#[inline]
pub fn num_nodes() -> usize {
    topology().num_nodes()
}

// ── Detection ────────────────────────────────────────────────────────

fn detect_topology() -> NumaTopology {
    #[cfg(target_os = "linux")]
    {
        if let Some(topo) = detect_sysfs_topology() {
            return topo;
        }
    }
    // Fallback: single node with all CPUs
    fallback_topology()
}

fn fallback_topology() -> NumaTopology {
    let ncpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    NumaTopology {
        nodes: vec![NumaNode {
            id: 0,
            cpus: (0..ncpus).collect(),
            mem_total: 0,
            l3_size: crate::cache_params::l3_size(),
        }],
        distances: vec![],
    }
}

#[cfg(target_os = "linux")]
fn detect_sysfs_topology() -> Option<NumaTopology> {
    use std::fs;
    use std::path::Path;

    let node_base = Path::new("/sys/devices/system/node");
    if !node_base.exists() {
        return None;
    }

    let mut nodes = Vec::new();
    let mut max_node_id = 0usize;

    // Enumerate nodeN directories
    let entries = fs::read_dir(node_base).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(id_str) = name_str.strip_prefix("node") {
            if let Ok(id) = id_str.parse::<usize>() {
                let node_path = node_base.join(&name);

                // Parse CPU list
                let cpulist_path = node_path.join("cpulist");
                let cpus = parse_cpulist(
                    &fs::read_to_string(cpulist_path).unwrap_or_default(),
                );
                if cpus.is_empty() {
                    continue; // Skip nodes with no CPUs (memory-only nodes)
                }

                // Parse memory total
                let meminfo_path = node_path.join("meminfo");
                let mem_total = parse_node_memtotal(
                    &fs::read_to_string(meminfo_path).unwrap_or_default(),
                );

                // Detect per-node L3 cache size from first CPU in this node
                let l3_size = if let Some(&first_cpu) = cpus.first() {
                    detect_cpu_l3(first_cpu)
                } else {
                    0
                };

                if id > max_node_id {
                    max_node_id = id;
                }

                nodes.push(NumaNode {
                    id,
                    cpus,
                    mem_total,
                    l3_size,
                });
            }
        }
    }

    if nodes.is_empty() {
        return None;
    }

    // Sort by node ID
    nodes.sort_by_key(|n| n.id);

    // Parse distance matrix
    let distances = parse_distances(node_base, max_node_id + 1);

    Some(NumaTopology { nodes, distances })
}

/// Parse a CPU list string like "0-3,8-11" into a Vec of CPU IDs.
fn parse_cpulist(s: &str) -> Vec<usize> {
    let s = s.trim();
    if s.is_empty() {
        return Vec::new();
    }
    let mut cpus = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(s..=e);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Parse MemTotal from node meminfo.
/// Format: "Node 0 MemTotal:       131817696 kB"
fn parse_node_memtotal(s: &str) -> usize {
    for line in s.lines() {
        if line.contains("MemTotal") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(kb) = parts[3].parse::<usize>() {
                    return kb * 1024; // Convert KB to bytes
                }
            }
        }
    }
    0
}

/// Detect L3 cache size for a specific CPU from sysfs.
#[cfg(target_os = "linux")]
fn detect_cpu_l3(cpu: usize) -> usize {
    use std::fs;
    for idx in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{idx}");
        let level = fs::read_to_string(format!("{base}/level"))
            .unwrap_or_default();
        let ctype = fs::read_to_string(format!("{base}/type"))
            .unwrap_or_default();
        let size_str = fs::read_to_string(format!("{base}/size"))
            .unwrap_or_default();

        if level.trim() == "3" && ctype.trim() == "Unified" {
            let size_str = size_str.trim();
            if let Some(kb) = size_str.strip_suffix('K') {
                return kb.parse::<usize>().unwrap_or(0) * 1024;
            } else if let Some(mb) = size_str.strip_suffix('M') {
                return mb.parse::<usize>().unwrap_or(0) * 1024 * 1024;
            }
        }
    }
    0
}

/// Parse the NUMA distance matrix from /sys/devices/system/node/nodeN/distance.
#[cfg(target_os = "linux")]
fn parse_distances(node_base: &std::path::Path, num_nodes: usize) -> Vec<Vec<u32>> {
    use std::fs;
    let mut distances = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let dist_path = node_base.join(format!("node{i}/distance"));
        let dist_str = fs::read_to_string(dist_path).unwrap_or_default();
        let row: Vec<u32> = dist_str
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if row.len() == num_nodes {
            distances.push(row);
        } else {
            return vec![]; // Incomplete, return empty
        }
    }
    distances
}

// ── Thread affinity ──────────────────────────────────────────────────

/// Pin the calling thread to a specific set of CPUs.
/// Uses `sched_setaffinity` on Linux.
/// Returns Ok(()) on success, Err on failure or unsupported platform.
#[cfg(target_os = "linux")]
pub fn pin_thread_to_cpus(cpus: &[usize]) -> Result<(), i32> {
    if cpus.is_empty() {
        return Ok(());
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        let ret = libc::sched_setaffinity(
            0, // current thread
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
        if ret == 0 {
            Ok(())
        } else {
            Err(*libc::__errno_location())
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to_cpus(_cpus: &[usize]) -> Result<(), i32> {
    Ok(()) // No-op on non-Linux
}

/// Pin the calling thread to all CPUs of a specific NUMA node.
pub fn pin_to_node(node_id: usize) -> Result<(), i32> {
    let topo = topology();
    if let Some(node) = topo.nodes.iter().find(|n| n.id == node_id) {
        pin_thread_to_cpus(&node.cpus)
    } else {
        Err(-1)
    }
}

// ── NUMA-aware memory policy ─────────────────────────────────────────

/// Memory policy constants for mbind().
#[cfg(target_os = "linux")]
mod mempolicy {
    pub const MPOL_DEFAULT: i32 = 0;
    pub const MPOL_PREFERRED: i32 = 1;
    pub const MPOL_BIND: i32 = 2;
    pub const MPOL_INTERLEAVE: i32 = 3;
    // Flags
    pub const MPOL_MF_STRICT: u32 = 1;
    pub const MPOL_MF_MOVE: u32 = 2;
}

/// Bind a memory region to a specific NUMA node using mbind().
/// The memory must already be allocated (e.g., via mmap or Vec).
/// This sets the NUMA policy so pages are allocated on the specified node.
#[cfg(target_os = "linux")]
pub fn mbind_to_node(ptr: *mut u8, len: usize, node_id: usize) -> Result<(), i32> {
    if len == 0 || !topology().is_multi_node() {
        return Ok(());
    }
    let max_node = topology().nodes.iter().map(|n| n.id).max().unwrap_or(0);
    let mask_len = (max_node / 8) + 1; // bytes needed
    let mut nodemask = vec![0u8; mask_len.max(8)]; // at least 8 bytes
    nodemask[node_id / 8] |= 1 << (node_id % 8);

    unsafe {
        let ret = libc::syscall(
            libc::SYS_mbind,
            ptr as libc::c_ulong,
            len as libc::c_ulong,
            mempolicy::MPOL_BIND as libc::c_int,
            nodemask.as_ptr() as libc::c_ulong,
            (max_node + 2) as libc::c_ulong, // maxnode (1-indexed + 1)
            (mempolicy::MPOL_MF_STRICT | mempolicy::MPOL_MF_MOVE) as libc::c_ulong,
        );
        if ret == 0 {
            Ok(())
        } else {
            Err(*libc::__errno_location())
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn mbind_to_node(_ptr: *mut u8, _len: usize, _node_id: usize) -> Result<(), i32> {
    Ok(())
}

/// Set interleaved memory policy for a region (pages round-robin across all nodes).
/// Good for shared read-only data like packed B panels.
#[cfg(target_os = "linux")]
pub fn mbind_interleave(ptr: *mut u8, len: usize) -> Result<(), i32> {
    if len == 0 || !topology().is_multi_node() {
        return Ok(());
    }
    let max_node = topology().nodes.iter().map(|n| n.id).max().unwrap_or(0);
    let mask_len = (max_node / 8) + 1;
    let mut nodemask = vec![0xFFu8; mask_len.max(8)]; // all nodes
    // Clear bits beyond max_node
    let last_byte = max_node / 8;
    let last_bit = max_node % 8;
    if last_byte < nodemask.len() {
        nodemask[last_byte] = (1u8 << (last_bit + 1)) - 1;
        for b in &mut nodemask[last_byte + 1..] {
            *b = 0;
        }
    }

    unsafe {
        let ret = libc::syscall(
            libc::SYS_mbind,
            ptr as libc::c_ulong,
            len as libc::c_ulong,
            mempolicy::MPOL_INTERLEAVE as libc::c_int,
            nodemask.as_ptr() as libc::c_ulong,
            (max_node + 2) as libc::c_ulong,
            (mempolicy::MPOL_MF_STRICT | mempolicy::MPOL_MF_MOVE) as libc::c_ulong,
        );
        if ret == 0 {
            Ok(())
        } else {
            Err(*libc::__errno_location())
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn mbind_interleave(_ptr: *mut u8, _len: usize) -> Result<(), i32> {
    Ok(())
}

// ── NUMA-aware aligned allocation ────────────────────────────────────

/// A cache-line aligned buffer bound to a specific NUMA node.
/// On single-node systems, behaves identically to `AlignedVec`.
pub struct NumaAlignedVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
    node_id: Option<usize>, // None = default policy
}

// SAFETY: NumaAlignedVec owns its heap allocation exclusively (same invariants as
// AlignedVec). NUMA binding is a placement hint and does not affect memory safety.
unsafe impl<T: Send> Send for NumaAlignedVec<T> {}
unsafe impl<T: Sync> Sync for NumaAlignedVec<T> {}

impl<T> NumaAlignedVec<T> {
    const ALIGN: usize = 64;

    /// Create an empty buffer (no allocation).
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            node_id: None,
        }
    }

    /// Create an empty buffer targeted at a specific NUMA node.
    #[inline]
    pub fn on_node(node_id: usize) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
            node_id: Some(node_id),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize { self.cap }
    #[inline]
    pub fn len(&self) -> usize { self.len }
    #[inline]
    pub fn as_ptr(&self) -> *const T { self.ptr }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    /// Allocate at least `new_cap` elements, preserving existing data.
    /// If a NUMA node is set, binds the new allocation to that node.
    pub fn reserve(&mut self, new_cap: usize) {
        if new_cap <= self.cap {
            return;
        }
        let elem_size = std::mem::size_of::<T>();
        assert!(elem_size > 0, "ZST not supported");
        let byte_size = new_cap * elem_size;
        let layout = std::alloc::Layout::from_size_align(byte_size, Self::ALIGN)
            .expect("invalid layout");
        let new_ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        assert!(!new_ptr.is_null(), "allocation failed");

        // Apply NUMA binding if on a multi-node system
        if let Some(node) = self.node_id {
            if let Err(e) = mbind_to_node(new_ptr as *mut u8, byte_size, node) {
                eprintln!("[gllm-kernels] debug: NUMA mbind to node {node} failed: {e}");
            }
        }

        // Copy existing data
        if self.len > 0 && !self.ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);
            }
        }
        self.dealloc();
        self.ptr = new_ptr;
        self.cap = new_cap;
    }

    /// Set length without initialization. Caller must ensure elements are valid.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.len = len;
    }

    fn dealloc(&mut self) {
        if !self.ptr.is_null() && self.cap > 0 {
            let elem_size = std::mem::size_of::<T>();
            let byte_size = self.cap * elem_size;
            let layout = std::alloc::Layout::from_size_align(byte_size, Self::ALIGN)
                .expect("invalid layout");
            unsafe {
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

impl<T> Drop for NumaAlignedVec<T> {
    fn drop(&mut self) {
        self.dealloc();
    }
}

impl<T> Default for NumaAlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Per-node thread pools ────────────────────────────────────────────

/// Per-NUMA-node rayon thread pools.
/// Each pool's threads are pinned to the CPUs of that node.
static NODE_POOLS: OnceLock<Vec<rayon::ThreadPool>> = OnceLock::new();

/// Get or create per-node thread pools.
/// On single-node systems, returns a single pool (the global rayon pool is used instead).
pub fn node_pools() -> &'static Vec<rayon::ThreadPool> {
    NODE_POOLS.get_or_init(|| {
        let topo = topology();
        if !topo.is_multi_node() {
            return vec![];
        }
        topo.nodes
            .iter()
            .map(|node| {
                let cpus = node.cpus.clone();
                let node_id = node.id;
                rayon::ThreadPoolBuilder::new()
                    .num_threads(cpus.len())
                    .start_handler(move |_thread_idx| {
                        if let Err(e) = pin_to_node(node_id) {
                            eprintln!(
                                "[gllm-kernels] debug: pin thread to NUMA node {node_id} failed: {e}"
                            );
                        }
                    })
                    .thread_name(move |idx| format!("numa{node_id}-worker{idx}"))
                    .build()
                    .unwrap_or_else(|e| panic!("failed to build thread pool for NUMA node {}: {e}", node.id))
            })
            .collect()
    })
}

/// Execute a closure on a specific NUMA node's thread pool.
/// On single-node systems, executes on the global rayon pool.
pub fn on_node<F, R>(node_id: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let pools = node_pools();
    if pools.is_empty() || node_id >= pools.len() {
        // Single-node or invalid node: use global pool
        f()
    } else {
        pools[node_id].install(f)
    }
}

/// Partition a range [0, total) across NUMA nodes proportionally to their CPU count.
/// Returns Vec<(node_id, start, end)> for each node's portion.
pub fn partition_by_nodes(total: usize, alignment: usize) -> Vec<(usize, usize, usize)> {
    let topo = topology();
    if !topo.is_multi_node() || total == 0 {
        return vec![(0, 0, total)];
    }

    let total_cpus = topo.total_cpus();
    if total_cpus == 0 {
        return vec![(0, 0, total)];
    }

    let mut partitions = Vec::with_capacity(topo.nodes.len());
    let mut offset = 0usize;

    for (i, node) in topo.nodes.iter().enumerate() {
        let is_last = i == topo.nodes.len() - 1;
        let share = if is_last {
            total - offset
        } else {
            let raw = (total * node.cpus.len()) / total_cpus;
            // Round down to alignment
            (raw / alignment) * alignment
        };
        if share > 0 {
            partitions.push((node.id, offset, offset + share));
            offset += share;
        }
    }

    // If rounding left nothing for some nodes, merge into last
    if partitions.is_empty() {
        partitions.push((0, 0, total));
    }

    partitions
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpulist() {
        assert_eq!(parse_cpulist("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0-3,8-11"), vec![0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(parse_cpulist("5"), vec![5]);
        assert_eq!(parse_cpulist(""), Vec::<usize>::new());
        assert_eq!(parse_cpulist("0,2,4"), vec![0, 2, 4]);
    }

    #[test]
    fn test_parse_memtotal() {
        let input = "Node 0 MemTotal:       131817696 kB\nNode 0 MemFree:         9867972 kB\n";
        assert_eq!(parse_node_memtotal(input), 131817696 * 1024);
    }

    #[test]
    fn test_topology_detection() {
        let topo = topology();
        assert!(!topo.nodes.is_empty(), "must have at least one node");
        assert!(topo.total_cpus() > 0, "must have at least one CPU");
        eprintln!("NUMA topology: {} nodes, {} total CPUs",
            topo.num_nodes(), topo.total_cpus());
        for node in &topo.nodes {
            eprintln!("  Node {}: {} CPUs {:?}, mem={} MB, L3={} KB",
                node.id, node.cpus.len(), &node.cpus,
                node.mem_total / (1024 * 1024),
                node.l3_size / 1024);
        }
        if !topo.distances.is_empty() {
            eprintln!("  Distance matrix: {:?}", topo.distances);
        }
    }

    #[test]
    fn test_partition_single_node() {
        // On this single-node system, partition should return one chunk
        let parts = partition_by_nodes(1024, 6);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], (0, 0, 1024));
    }

    #[test]
    fn test_pin_thread() {
        // Should succeed on Linux (pin to all CPUs = no restriction)
        let topo = topology();
        let all_cpus: Vec<usize> = topo.nodes.iter().flat_map(|n| n.cpus.iter().copied()).collect();
        let result = pin_thread_to_cpus(&all_cpus);
        assert!(result.is_ok(), "pin_thread_to_cpus failed: {:?}", result);
    }

    #[test]
    fn test_numa_aligned_vec() {
        let mut v = NumaAlignedVec::<f32>::on_node(0);
        v.reserve(1024);
        assert!(v.capacity() >= 1024);
        assert_eq!(v.as_ptr() as usize % 64, 0, "not 64-byte aligned");
        unsafe {
            v.set_len(1024);
            for i in 0..1024 {
                *v.as_mut_ptr().add(i) = i as f32;
            }
            for i in 0..1024 {
                assert_eq!(*v.as_ptr().add(i), i as f32);
            }
        }
    }
}
