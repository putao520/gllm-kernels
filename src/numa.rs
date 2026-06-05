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

    /// Compute NC blocking for a single NUMA node's L3 cache.
    ///
    /// On multi-node systems, the global L3 size overstates what a single socket
    /// can access locally. This method computes an NC that fits within one node's
    /// L3, aligned to NR for microkernel compatibility.
    ///
    /// Returns `None` on single-node systems (use global NC instead).
    pub fn nc_for_node_l3(&self, n: usize, nr: usize) -> Option<usize> {
        if !self.is_multi_node() || nr == 0 {
            return None;
        }
        // Use first node's L3 as representative (nodes are typically symmetric)
        let node_l3 = self.nodes.first().map(|n| n.l3_size).unwrap_or(0);
        if node_l3 == 0 {
            return None;
        }
        // NC = floor(node_l3 / (nr * 4)) / nr * nr  (aligned to NR, f32 elements)
        let nc = (node_l3 / (nr * 4)) / nr * nr;
        Some(nc.min(n).max(nr))
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
        // SAFETY: ALIGN is a power of 2 (64) and byte_size > 0 (asserted above via elem_size > 0)
        let layout = std::alloc::Layout::from_size_align(byte_size, Self::ALIGN)
            .expect("ALIGN is a valid power-of-2 constant");
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
            // SAFETY: reconstructing the same layout used for allocation (ALIGN is constant)
            let layout = std::alloc::Layout::from_size_align(byte_size, Self::ALIGN)
                .expect("ALIGN is a valid power-of-2 constant");
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

    #[test]
    fn test_nc_for_node_l3_single_node() {
        let topo = NumaTopology {
            nodes: vec![NumaNode { id: 0, cpus: vec![0, 1], mem_total: 0, l3_size: 8 * 1024 * 1024 }],
            distances: vec![],
        };
        // Single node: should return None
        assert_eq!(topo.nc_for_node_l3(4096, 16), None);
    }

    #[test]
    fn test_nc_for_node_l3_multi_node() {
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1], mem_total: 0, l3_size: 8 * 1024 * 1024 },
                NumaNode { id: 1, cpus: vec![2, 3], mem_total: 0, l3_size: 8 * 1024 * 1024 },
            ],
            distances: vec![vec![10, 21], vec![21, 10]],
        };
        let nc = topo.nc_for_node_l3(4096, 16).unwrap();
        assert_eq!(nc % 16, 0, "NC {} not aligned to NR=16", nc);
        assert!(nc <= 4096);
        assert!(nc >= 16);
    }

    // @trace TEST-NUMA-09 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_cpu_to_node_found_and_missing() {
        // Arrange: construct a two-node topology
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 2, 4], mem_total: 0, l3_size: 0 },
                NumaNode { id: 1, cpus: vec![1, 3, 5], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act & Assert: CPUs present
        assert_eq!(topo.cpu_to_node(0), Some(0));
        assert_eq!(topo.cpu_to_node(1), Some(1));
        assert_eq!(topo.cpu_to_node(5), Some(1));

        // Act & Assert: CPU not in any node
        assert_eq!(topo.cpu_to_node(99), None);
    }

    // @trace TEST-NUMA-10 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_node_l3_size_known_and_fallback() {
        // Arrange: two nodes, node 0 has known L3, node 1 has l3_size=0
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 6 * 1024 * 1024 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act & Assert: node 0 returns its own L3
        assert_eq!(topo.node_l3_size(0), 6 * 1024 * 1024);

        // Act & Assert: node 1 with l3_size=0 falls back to cache_params::l3_size()
        let fallback = crate::cache_params::l3_size();
        assert!(fallback > 0, "cache_params::l3_size() should be > 0");
        assert_eq!(topo.node_l3_size(1), fallback);

        // Act & Assert: out-of-range node_id also falls back
        assert_eq!(topo.node_l3_size(5), fallback);
    }

    // @trace TEST-NUMA-11 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_nc_for_node_l3_edge_cases() {
        // Arrange: multi-node topology for edge-case testing
        let multi_topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 32 * 1024 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 32 * 1024 },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        };

        // Act: nr=0 => None (guard against division by zero)
        assert_eq!(multi_topo.nc_for_node_l3(1024, 0), None);

        // Act: L3 so small that floor yields 0 => clamped to nr
        let nc_tiny = multi_topo.nc_for_node_l3(1024, 16).unwrap();
        assert!(nc_tiny >= 16, "NC should be clamped to at least nr");

        // Arrange: multi-node but l3_size=0
        let zero_l3_topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 0 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act: zero L3 => None
        assert_eq!(zero_l3_topo.nc_for_node_l3(4096, 16), None);
    }

    // @trace TEST-NUMA-12 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_topology_accessors_on_constructed() {
        // Arrange
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1, 2], mem_total: 16 * 1024 * 1024 * 1024, l3_size: 8 * 1024 * 1024 },
                NumaNode { id: 1, cpus: vec![3, 4], mem_total: 8 * 1024 * 1024 * 1024, l3_size: 4 * 1024 * 1024 },
            ],
            distances: vec![vec![10, 21], vec![21, 10]],
        };

        // Act & Assert
        assert_eq!(topo.num_nodes(), 2);
        assert!(topo.is_multi_node());
        assert_eq!(topo.total_cpus(), 5);
    }

    // @trace TEST-NUMA-13 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_single_node_topology_accessors() {
        // Arrange
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act & Assert
        assert_eq!(topo.num_nodes(), 1);
        assert!(!topo.is_multi_node());
        assert_eq!(topo.total_cpus(), 2);
    }

    // @trace TEST-NUMA-14 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_cpulist_whitespace_and_invalid() {
        // Arrange & Act & Assert: leading/trailing whitespace
        assert_eq!(parse_cpulist("  0-2  "), vec![0, 1, 2]);

        // Arrange & Act & Assert: whitespace around entries
        assert_eq!(parse_cpulist(" 0 , 2 , 4 "), vec![0, 2, 4]);

        // Arrange & Act & Assert: single range with surrounding spaces
        assert_eq!(parse_cpulist("  3-5  "), vec![3, 4, 5]);

        // Arrange & Act & Assert: invalid token ignored (not a number)
        assert_eq!(parse_cpulist("abc"), Vec::<usize>::new());

        // Arrange & Act & Assert: mixed valid and invalid — invalid parts skipped
        assert_eq!(parse_cpulist("0-1,abc,4"), vec![0, 1, 4]);
    }

    // @trace TEST-NUMA-15 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_memtotal_missing_and_malformed() {
        // Arrange: input with no MemTotal line
        let no_total = "Node 0 MemFree:         9867972 kB\n";
        // Act & Assert
        assert_eq!(parse_node_memtotal(no_total), 0);

        // Arrange: MemTotal line with too few tokens
        let short_line = "Node 0 MemTotal:\n";
        // Act & Assert
        assert_eq!(parse_node_memtotal(short_line), 0);

        // Arrange: MemTotal line with non-numeric value
        let bad_num = "Node 0 MemTotal:       xxx kB\n";
        // Act & Assert
        assert_eq!(parse_node_memtotal(bad_num), 0);

        // Arrange: empty string
        // Act & Assert
        assert_eq!(parse_node_memtotal(""), 0);
    }

    // @trace TEST-NUMA-16 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_pin_thread_to_cpus_empty_slice() {
        // Arrange: empty CPU list
        let cpus: Vec<usize> = vec![];

        // Act
        let result = pin_thread_to_cpus(&cpus);

        // Assert: empty slice is an early-return Ok
        assert!(result.is_ok());
    }

    // @trace TEST-NUMA-17 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_pin_to_node_invalid_id() {
        // Arrange: use an implausibly large node_id that cannot exist
        let invalid_node = 9999;

        // Act
        let result = pin_to_node(invalid_node);

        // Assert: should return Err(-1) for nonexistent node
        assert_eq!(result, Err(-1));
    }

    // @trace TEST-NUMA-18 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_new_and_default() {
        // Arrange & Act
        let v = NumaAlignedVec::<f32>::new();

        // Assert: zero state
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert!(v.as_ptr().is_null());

        // Act: Default trait
        let v_default = NumaAlignedVec::<u8>::default();

        // Assert: same zero state
        assert_eq!(v_default.len(), 0);
        assert_eq!(v_default.capacity(), 0);
    }

    // @trace TEST-NUMA-19 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_reserve_noop_when_cap_sufficient() {
        // Arrange: pre-reserve 512 elements
        let mut v = NumaAlignedVec::<f64>::new();
        v.reserve(512);
        let cap_before = v.capacity();
        assert!(cap_before >= 512);

        // Act: request less than current capacity
        v.reserve(256);

        // Assert: capacity unchanged
        assert_eq!(v.capacity(), cap_before);

        // Act: request exactly current capacity
        v.reserve(cap_before);

        // Assert: still unchanged
        assert_eq!(v.capacity(), cap_before);
    }

    // @trace TEST-NUMA-20 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_reserve_preserves_data() {
        // Arrange: allocate and write initial data
        let mut v = NumaAlignedVec::<u64>::new();
        v.reserve(4);
        unsafe {
            v.set_len(4);
            for i in 0..4 {
                *v.as_mut_ptr().add(i) = (i + 100) as u64;
            }
        }

        // Act: grow the buffer
        v.reserve(64);

        // Assert: old data preserved
        unsafe {
            for i in 0..4 {
                assert_eq!(*v.as_ptr().add(i), (i + 100) as u64);
            }
        }
        assert!(v.capacity() >= 64);
        assert_eq!(v.len(), 4);
    }

    // @trace TEST-NUMA-21 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_partition_by_nodes_zero_total() {
        // Arrange & Act
        let parts = partition_by_nodes(0, 64);

        // Assert: should return a single partition covering [0, 0)
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], (0, 0, 0));
    }

    // @trace TEST-NUMA-22 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_nc_for_node_l3_clamps_to_n() {
        // Arrange: n=8 (very small), nr=4, large L3 so computed nc >> n
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 32 * 1024 * 1024 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 32 * 1024 * 1024 },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        };

        // Act
        let nc = topo.nc_for_node_l3(8, 4).unwrap();

        // Assert: nc should be clamped to at most n=8, and at least nr=4
        assert!(nc <= 8);
        assert!(nc >= 4);
        assert_eq!(nc % 4, 0, "NC should be aligned to NR");
    }

    // @trace TEST-NUMA-23 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_node_clone_and_debug() {
        // Arrange
        let node = NumaNode {
            id: 3,
            cpus: vec![10, 11, 12],
            mem_total: 64 * 1024 * 1024 * 1024,
            l3_size: 16 * 1024 * 1024,
        };

        // Act: Clone
        let cloned = node.clone();

        // Assert: fields match
        assert_eq!(cloned.id, 3);
        assert_eq!(cloned.cpus, vec![10, 11, 12]);
        assert_eq!(cloned.mem_total, 64 * 1024 * 1024 * 1024);
        assert_eq!(cloned.l3_size, 16 * 1024 * 1024);

        // Act: Debug format
        let debug_str = format!("{:?}", node);

        // Assert: contains key fields
        assert!(debug_str.contains("id: 3"));
        assert!(debug_str.contains("mem_total:"));
    }

    // @trace TEST-NUMA-24 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_topology_clone_and_debug() {
        // Arrange
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 1024, l3_size: 2048 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 4096, l3_size: 8192 },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        };

        // Act: Clone
        let cloned = topo.clone();

        // Assert: equal content
        assert_eq!(cloned.num_nodes(), 2);
        assert_eq!(cloned.nodes[0].id, 0);
        assert_eq!(cloned.nodes[1].id, 1);
        assert_eq!(cloned.distances, vec![vec![10, 20], vec![20, 10]]);

        // Act: Debug format
        let debug_str = format!("{:?}", topo);

        // Assert: contains nodes and distances
        assert!(debug_str.contains("nodes:"));
        assert!(debug_str.contains("distances:"));
    }

    // @trace TEST-NUMA-25 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_on_node_constructor() {
        // Arrange & Act
        let v = NumaAlignedVec::<f32>::on_node(2);

        // Assert: zero-length but node_id is set
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert!(v.as_ptr().is_null());

        // Act: reserve and verify alignment still works
        let mut v = v;
        v.reserve(256);

        // Assert: properly allocated and aligned
        assert!(v.capacity() >= 256);
        assert_eq!(v.as_ptr() as usize % 64, 0, "must be 64-byte aligned");
    }

    // @trace TEST-NUMA-26 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_u8_type() {
        // Arrange
        let mut v = NumaAlignedVec::<u8>::new();

        // Act: reserve and write bytes
        v.reserve(128);
        unsafe {
            v.set_len(128);
            for i in 0..128 {
                *v.as_mut_ptr().add(i) = i as u8;
            }
        }

        // Assert: read back matches
        unsafe {
            for i in 0..128 {
                assert_eq!(*v.as_ptr().add(i), i as u8);
            }
        }
        assert_eq!(v.len(), 128);
    }

    // @trace TEST-NUMA-27 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_drop_no_double_free() {
        // Arrange: create and fill a buffer
        let mut v = NumaAlignedVec::<u64>::new();
        v.reserve(16);
        unsafe {
            v.set_len(16);
            for i in 0..16 {
                *v.as_mut_ptr().add(i) = i as u64;
            }
        }

        // Act: drop by going out of scope — this test passes if no double-free/UB
        drop(v);

        // Assert: (no assertion needed — passes if no panic/UB under sanitizers)
    }

    // @trace TEST-NUMA-28 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_cpu_to_node_first_match_wins() {
        // Arrange: contrived topology where CPU 2 appears in both nodes
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 2], mem_total: 0, l3_size: 0 },
                NumaNode { id: 5, cpus: vec![2, 3], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act & Assert: first node containing CPU 2 wins (node 0)
        assert_eq!(topo.cpu_to_node(2), Some(0));

        // Act & Assert: CPU 3 is only in node 5
        assert_eq!(topo.cpu_to_node(3), Some(5));
    }

    // @trace TEST-NUMA-29 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_total_cpus_empty_nodes() {
        // Arrange: topology with a single node but empty CPU list
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act & Assert
        assert_eq!(topo.total_cpus(), 0);
        assert_eq!(topo.num_nodes(), 1);
        assert!(!topo.is_multi_node());
    }

    // @trace TEST-NUMA-30 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_nc_for_node_l3_exact_computation() {
        // Arrange: L3 = 8192 bytes, nr = 4
        // nc = (8192 / (4*4)) / 4 * 4 = 512 / 4 * 4 = 512
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 8192 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 8192 },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        };

        // Act
        let nc = topo.nc_for_node_l3(1024, 4).unwrap();

        // Assert: 512, aligned to nr=4
        assert_eq!(nc, 512);
        assert_eq!(nc % 4, 0);
    }

    // @trace TEST-NUMA-31 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_node_l3_size_single_node_known() {
        // Arrange: single node with known L3
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1], mem_total: 0, l3_size: 12 * 1024 * 1024 },
            ],
            distances: vec![],
        };

        // Act & Assert: node 0 has l3_size > 0, so it should return that directly
        assert_eq!(topo.node_l3_size(0), 12 * 1024 * 1024);

        // Act & Assert: node 1 is out of range, falls back to cache_params
        let fallback = crate::cache_params::l3_size();
        assert_eq!(topo.node_l3_size(1), fallback);
    }

    // @trace TEST-NUMA-32 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_cpulist_single_element_and_multi_range() {
        // Arrange & Act & Assert: single element
        assert_eq!(parse_cpulist("7"), vec![7]);

        // Arrange & Act & Assert: three disjoint ranges
        assert_eq!(
            parse_cpulist("0-1,10-11,20-21"),
            vec![0, 1, 10, 11, 20, 21],
        );

        // Arrange & Act & Assert: overlapping ranges (parser does not dedup)
        let result = parse_cpulist("0-3,2-5");
        assert_eq!(result, vec![0, 1, 2, 3, 2, 3, 4, 5]);
    }

    // @trace TEST-NUMA-33 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_empty_topology_zero_nodes() {
        // Arrange: topology with no nodes at all
        let topo = NumaTopology {
            nodes: vec![],
            distances: vec![],
        };

        // Act & Assert: accessors on empty topology
        assert_eq!(topo.num_nodes(), 0);
        assert!(!topo.is_multi_node());
        assert_eq!(topo.total_cpus(), 0);

        // Act & Assert: cpu_to_node always returns None
        assert_eq!(topo.cpu_to_node(0), None);
    }

    // @trace TEST-NUMA-34 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_distance_matrix_four_node_symmetric() {
        // Arrange: 4-node topology with realistic distance values
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1], mem_total: 0, l3_size: 0 },
                NumaNode { id: 1, cpus: vec![2, 3], mem_total: 0, l3_size: 0 },
                NumaNode { id: 2, cpus: vec![4, 5], mem_total: 0, l3_size: 0 },
                NumaNode { id: 3, cpus: vec![6, 7], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![
                vec![10, 21, 28, 28],
                vec![21, 10, 28, 28],
                vec![28, 28, 10, 21],
                vec![28, 28, 21, 10],
            ],
        };

        // Act & Assert: diagonal is always 10 (local)
        for i in 0..4 {
            assert_eq!(topo.distances[i][i], 10, "node {i} self-distance should be 10");
        }

        // Act & Assert: symmetry — distances[i][j] == distances[j][i]
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    topo.distances[i][j], topo.distances[j][i],
                    "asymmetry at [{i}][{j}]"
                );
            }
        }

        // Act & Assert: matrix dimensions match node count
        assert_eq!(topo.distances.len(), 4);
        for row in &topo.distances {
            assert_eq!(row.len(), 4);
        }
    }

    // @trace TEST-NUMA-35 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_node_debug_format_all_fields() {
        // Arrange
        let node = NumaNode {
            id: 7,
            cpus: vec![14, 15, 16],
            mem_total: 32 * 1024 * 1024 * 1024,
            l3_size: 24 * 1024 * 1024,
        };

        // Act
        let debug = format!("{:?}", node);

        // Assert: all four fields appear in Debug output
        assert!(debug.contains("id: 7"), "missing id field");
        assert!(debug.contains("cpus"), "missing cpus field");
        assert!(debug.contains("mem_total"), "missing mem_total field");
        assert!(debug.contains("l3_size"), "missing l3_size field");
    }

    // @trace TEST-NUMA-36 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_topology_debug_format_empty_vs_populated() {
        // Arrange: empty topology
        let empty = NumaTopology { nodes: vec![], distances: vec![] };
        let empty_debug = format!("{:?}", empty);

        // Assert: Debug works on empty state without panic
        assert!(empty_debug.contains("nodes: []"));
        assert!(empty_debug.contains("distances: []"));

        // Arrange: populated topology
        let populated = NumaTopology {
            nodes: vec![NumaNode { id: 0, cpus: vec![0], mem_total: 4096, l3_size: 1024 }],
            distances: vec![vec![10]],
        };
        let pop_debug = format!("{:?}", populated);

        // Assert: contains key structural markers
        assert!(pop_debug.contains("distances:"));
        assert!(pop_debug.contains("10"));
    }

    // @trace TEST-NUMA-37 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_default_trait() {
        // Arrange & Act: construct via Default trait
        let v: NumaAlignedVec<i32> = NumaAlignedVec::default();

        // Assert: equivalent to ::new()
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert!(v.as_ptr().is_null());
    }

    // @trace TEST-NUMA-38 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_memtotal_multiple_nodes() {
        // Arrange: meminfo from a 2-node system, requesting node 1's total
        let input_node1 = concat!(
            "Node 1 MemTotal:       65536000 kB\n",
            "Node 1 MemFree:        12345000 kB\n",
        );

        // Act
        let result = parse_node_memtotal(input_node1);

        // Assert: parses the correct line for node 1
        assert_eq!(result, 65536000 * 1024);

        // Arrange: input with MemTotal in unusual position (second line)
        let multi_line = concat!(
            "Node 0 MemFree:        100 kB\n",
            "Node 0 MemTotal:       999 kB\n",
        );

        // Act & Assert: still finds the MemTotal line
        assert_eq!(parse_node_memtotal(multi_line), 999 * 1024);
    }

    // @trace TEST-NUMA-39 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_cpu_to_node_single_cpu_per_node() {
        // Arrange: 3 nodes, 1 CPU each
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 0 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 0 },
                NumaNode { id: 2, cpus: vec![2], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![vec![10, 20, 30], vec![20, 10, 30], vec![30, 30, 10]],
        };

        // Act & Assert: each CPU maps to its expected node
        assert_eq!(topo.cpu_to_node(0), Some(0));
        assert_eq!(topo.cpu_to_node(1), Some(1));
        assert_eq!(topo.cpu_to_node(2), Some(2));

        // Act & Assert: non-existent CPU returns None
        assert_eq!(topo.cpu_to_node(3), None);
        assert_eq!(topo.cpu_to_node(usize::MAX), None);
    }

    // @trace TEST-NUMA-40 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_nc_for_node_l3_nr_equals_n() {
        // Arrange: n=4, nr=4, L3 large enough that nc would exceed n
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 1024 * 1024 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 1024 * 1024 },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        };

        // Act: when computed nc > n, it should clamp to n=4
        let nc = topo.nc_for_node_l3(4, 4).unwrap();

        // Assert: nc == 4 (equal to both n and nr), properly aligned
        assert_eq!(nc, 4);
        assert_eq!(nc % 4, 0);
    }

    // @trace TEST-NUMA-41 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_multiple_reserve_grow_cycles() {
        // Arrange: start empty
        let mut v = NumaAlignedVec::<u32>::new();

        // Act: first allocation
        v.reserve(16);
        assert!(v.capacity() >= 16);
        let ptr1 = v.as_ptr();

        // Act: write data
        unsafe {
            v.set_len(16);
            for i in 0..16 {
                *v.as_mut_ptr().add(i) = i as u32;
            }
        }

        // Act: grow to larger size
        v.reserve(128);
        assert!(v.capacity() >= 128);

        // Assert: data survives reallocation
        unsafe {
            for i in 0..16 {
                assert_eq!(*v.as_ptr().add(i), i as u32);
            }
        }

        // Assert: pointer may have changed (reallocation occurred)
        // but alignment is maintained
        assert_eq!(v.as_ptr() as usize % 64, 0);
    }

    // @trace TEST-NUMA-42 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_partition_by_nodes_non_zero_alignment_single_node() {
        // Arrange: any positive alignment on single-node system
        // Act
        let parts_a = partition_by_nodes(256, 1);
        let parts_b = partition_by_nodes(256, 128);

        // Assert: single-node always returns one partition covering [0, total)
        assert_eq!(parts_a, vec![(0, 0, 256)]);
        assert_eq!(parts_b, vec![(0, 0, 256)]);
    }

    // @trace TEST-NUMA-43 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_cpulist_large_range() {
        // Arrange: a range spanning many CPUs
        let input = "0-127";

        // Act
        let cpus = parse_cpulist(input);

        // Assert: 128 CPUs produced, first=0, last=127, all sequential
        assert_eq!(cpus.len(), 128);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[127], 127);
        for i in 0..128 {
            assert_eq!(cpus[i], i);
        }
    }

    // @trace TEST-NUMA-44 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_topology_non_contiguous_node_ids() {
        // Arrange: nodes with IDs 0, 3, 7 (non-contiguous)
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1], mem_total: 8 * 1024 * 1024 * 1024, l3_size: 4 * 1024 * 1024 },
                NumaNode { id: 3, cpus: vec![4, 5], mem_total: 16 * 1024 * 1024 * 1024, l3_size: 8 * 1024 * 1024 },
                NumaNode { id: 7, cpus: vec![8, 9], mem_total: 32 * 1024 * 1024 * 1024, l3_size: 12 * 1024 * 1024 },
            ],
            distances: vec![],
        };

        // Act & Assert: num_nodes counts by vector length, not max ID
        assert_eq!(topo.num_nodes(), 3);
        assert!(topo.is_multi_node());
        assert_eq!(topo.total_cpus(), 6);

        // Act & Assert: cpu_to_node returns the actual node ID, not index
        assert_eq!(topo.cpu_to_node(0), Some(0));
        assert_eq!(topo.cpu_to_node(4), Some(3));
        assert_eq!(topo.cpu_to_node(8), Some(7));
        assert_eq!(topo.cpu_to_node(2), None);
    }

    // @trace TEST-NUMA-45 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_nc_for_node_l3_uses_first_node_only() {
        // Arrange: asymmetric topology — node 0 has tiny L3, node 1 has large L3
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 256 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 64 * 1024 * 1024 },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        };

        // Act: nc_for_node_l3 should use node 0's L3 (first node)
        let nc = topo.nc_for_node_l3(4096, 4).unwrap();

        // Assert: computed from node 0's tiny 256-byte L3, not node 1's large L3
        // nc = (256 / (4*4)) / 4 * 4 = 16 / 4 * 4 = 16, clamped to >= nr=4
        assert_eq!(nc, 16);
        assert!(nc < 4096, "should be much smaller than n since node 0 L3 is tiny");
    }

    // @trace TEST-NUMA-46 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_aligned_vec_on_node_reserve_write_read() {
        // Arrange: create buffer targeting node 0
        let mut v = NumaAlignedVec::<u32>::on_node(0);

        // Act: reserve and populate with a known pattern
        v.reserve(512);
        unsafe {
            v.set_len(512);
            for i in 0..512 {
                *v.as_mut_ptr().add(i) = (i * 3 + 7) as u32;
            }
        }

        // Assert: all data reads back correctly
        unsafe {
            for i in 0..512 {
                assert_eq!(*v.as_ptr().add(i), (i * 3 + 7) as u32,
                    "data mismatch at index {}", i);
            }
        }
        assert!(v.capacity() >= 512);
        assert_eq!(v.as_ptr() as usize % 64, 0, "must be 64-byte aligned");
    }

    // @trace TEST-NUMA-47 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_distance_matrix_local_always_smallest() {
        // Arrange: 3-node topology with realistic distances
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0], mem_total: 0, l3_size: 0 },
                NumaNode { id: 1, cpus: vec![1], mem_total: 0, l3_size: 0 },
                NumaNode { id: 2, cpus: vec![2], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![
                vec![10, 21, 28],
                vec![21, 10, 25],
                vec![28, 25, 10],
            ],
        };

        // Act & Assert: local distance (diagonal) is strictly less than any remote
        for i in 0..topo.num_nodes() {
            let local = topo.distances[i][i];
            assert_eq!(local, 10, "local distance should be 10");
            for j in 0..topo.num_nodes() {
                if i != j {
                    assert!(
                        topo.distances[i][j] > local,
                        "remote distance [{}][{}]={} should exceed local {}",
                        i, j, topo.distances[i][j], local,
                    );
                }
            }
        }
    }

    // @trace TEST-NUMA-48 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_node_memtotal_extra_whitespace() {
        // Arrange: MemTotal line with extra spaces between fields
        let input = "Node 0 MemTotal:    2048000 kB\n";

        // Act
        let result = parse_node_memtotal(input);

        // Assert: split_whitespace handles arbitrary spacing, value converted to bytes
        assert_eq!(result, 2048000 * 1024);
    }

    // @trace TEST-NUMA-49 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_topology_varying_cpu_counts_per_node() {
        // Arrange: 3 nodes with asymmetric CPU counts (4, 2, 8)
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1, 2, 3], mem_total: 0, l3_size: 0 },
                NumaNode { id: 1, cpus: vec![4, 5], mem_total: 0, l3_size: 0 },
                NumaNode { id: 2, cpus: vec![6, 7, 8, 9, 10, 11, 12, 13], mem_total: 0, l3_size: 0 },
            ],
            distances: vec![],
        };

        // Act & Assert: total_cpus sums all nodes correctly
        assert_eq!(topo.total_cpus(), 4 + 2 + 8);
        assert_eq!(topo.num_nodes(), 3);
        assert!(topo.is_multi_node());

        // Assert: cpu_to_node maps correctly across asymmetric boundaries
        assert_eq!(topo.cpu_to_node(3), Some(0));
        assert_eq!(topo.cpu_to_node(4), Some(1));
        assert_eq!(topo.cpu_to_node(5), Some(1));
        assert_eq!(topo.cpu_to_node(6), Some(2));
        assert_eq!(topo.cpu_to_node(13), Some(2));
        assert_eq!(topo.cpu_to_node(14), None);
    }

    // @trace TEST-NUMA-50 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_numa_node_mem_total_kb_to_bytes_conversion() {
        // Arrange: node with a realistic memory value in GB range
        // 131072 MB = 134217728 MB = 134217728 * 1024 KB
        let mem_kb = 134217728usize;
        let node = NumaNode {
            id: 0,
            cpus: vec![0],
            mem_total: mem_kb * 1024, // simulate what parse_node_memtotal would produce
            l3_size: 0,
        };

        // Act & Assert: mem_total is in bytes (KB * 1024)
        assert_eq!(node.mem_total, mem_kb * 1024);
        // Verify it represents ~128 GB
        let mem_gb = node.mem_total / (1024 * 1024 * 1024);
        assert_eq!(mem_gb, 128);
    }

    // @trace TEST-NUMA-51 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_parse_cpulist_tab_and_mixed_whitespace() {
        // Arrange: cpulist with tab characters and trailing newline
        let tab_input = "0-2\t,5\t";
        // Act
        let result = parse_cpulist(tab_input);
        // Assert: trim handles outer whitespace, comma splits correctly
        assert_eq!(result, vec![0, 1, 2, 5]);

        // Arrange: cpulist with newline
        let newline_input = "0-1\n";
        // Act
        let result2 = parse_cpulist(newline_input);
        // Assert: newline is trimmed
        assert_eq!(result2, vec![0, 1]);
    }

    // @trace TEST-NUMA-52 [req:REQ-NUMA] [level:unit]
    #[test]
    fn test_empty_distance_matrix_single_node() {
        // Arrange: single-node topology with empty distances
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { id: 0, cpus: vec![0, 1, 2], mem_total: 4096, l3_size: 1024 },
            ],
            distances: vec![],
        };

        // Act & Assert: distances vector is empty on single-node
        assert!(topo.distances.is_empty());

        // Assert: other accessors still work correctly
        assert_eq!(topo.num_nodes(), 1);
        assert!(!topo.is_multi_node());
        assert_eq!(topo.total_cpus(), 3);
    }
}
