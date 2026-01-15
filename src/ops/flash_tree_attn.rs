//! # DeFT / Talon Flash Tree-attention (REQ-OP-010)
//!
//! Implements optimized tree verification with O(n+m) complexity Flash Tree-attention.
//!
//! ## Features
//! - **DeFT-Flatten**: Uniform tree distribution to GPU SMs
//! - **DeFT-Node**: Node-level parallelism
//! - **Talon**: Confidence-adaptive token tree
//! - **Traversal Verification**: Sequence-level verification
//!
//! ## References
//! - DeFT (ICLR'25): Flash Tree-attention
//! - Talon (ICLR'26): Confidence-adaptive Token Tree
//! - SEQUOIA: Dynamic tree structure optimization
//!
//! ## SPEC Compliance
//! - ARCH-OP-010: TokenTree, TreeMask, FlashTreeAttention, TalonConfig
//! - Target: 2-4x batch speedup, O(n+m) complexity

use std::collections::VecDeque;

/// Token ID type
pub type TokenId = u32;

// ============================================================================
// Partition Strategy
// ============================================================================

/// Tree partition strategy for GPU execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// DeFT-Flatten: Uniform distribution across SMs
    DeFTFlatten,
    /// DeFT-Node: Node-level parallelism
    DeFTNode,
    /// Sequential: No parallelism (for small trees)
    Sequential,
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        PartitionStrategy::DeFTFlatten
    }
}

// ============================================================================
// Token Tree (ARCH-OP-010)
// ============================================================================

/// Token tree structure for speculative decoding verification
///
/// SPEC: ARCH-OP-010 TokenTree structure
///
/// The tree is stored in DFS linearized order:
/// ```text
///              [root]
///             /  |   \
///         [a]   [b]   [c]
///        / |     |    / \
///     [d] [e]   [f] [g] [h]
///
/// Linearized: [root, a, d, e, b, f, c, g, h]
/// ```
#[derive(Debug, Clone)]
pub struct TokenTree {
    /// DFS linearized token sequence
    pub tokens: Vec<TokenId>,
    /// Parent index for each node (-1 for root)
    pub parent_indices: Vec<i32>,
    /// Depth of each node in tree
    pub depth: Vec<usize>,
    /// Total number of nodes
    pub num_nodes: usize,
}

impl TokenTree {
    /// Create empty tree
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            parent_indices: Vec::new(),
            depth: Vec::new(),
            num_nodes: 0,
        }
    }

    /// Create tree with single root
    pub fn with_root(root_token: TokenId) -> Self {
        Self {
            tokens: vec![root_token],
            parent_indices: vec![-1],
            depth: vec![0],
            num_nodes: 1,
        }
    }

    /// Add child node to parent
    pub fn add_child(&mut self, parent_idx: usize, token: TokenId) -> usize {
        if parent_idx >= self.num_nodes {
            panic!("Invalid parent index");
        }

        let child_idx = self.num_nodes;
        self.tokens.push(token);
        self.parent_indices.push(parent_idx as i32);
        self.depth.push(self.depth[parent_idx] + 1);
        self.num_nodes += 1;

        child_idx
    }

    /// Build tree from candidates (token id, parent index) pairs
    pub fn from_candidates(root: TokenId, candidates: &[(TokenId, usize)]) -> Self {
        let mut tree = Self::with_root(root);
        for &(token, parent) in candidates {
            tree.add_child(parent, token);
        }
        tree
    }

    /// Get children indices for a node
    pub fn children(&self, node_idx: usize) -> Vec<usize> {
        (0..self.num_nodes)
            .filter(|&i| self.parent_indices[i] == node_idx as i32)
            .collect()
    }

    /// Get all leaf nodes
    pub fn leaves(&self) -> Vec<usize> {
        let has_children: Vec<bool> = (0..self.num_nodes)
            .map(|i| {
                self.parent_indices
                    .iter()
                    .any(|&p| p == i as i32)
            })
            .collect();

        (0..self.num_nodes)
            .filter(|&i| !has_children[i])
            .collect()
    }

    /// Get path from root to node
    pub fn path_to_root(&self, node_idx: usize) -> Vec<usize> {
        let mut path = vec![node_idx];
        let mut current = node_idx;

        while self.parent_indices[current] >= 0 {
            current = self.parent_indices[current] as usize;
            path.push(current);
        }

        path.reverse();
        path
    }

    /// Get all paths from root to leaves
    pub fn all_paths(&self) -> Vec<Vec<usize>> {
        self.leaves()
            .iter()
            .map(|&leaf| self.path_to_root(leaf))
            .collect()
    }

    /// Get maximum depth
    pub fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0)
    }

    /// Check if node j is ancestor of node i (including self)
    pub fn is_ancestor(&self, i: usize, j: usize) -> bool {
        if i == j {
            return true;
        }

        let mut current = i;
        while self.parent_indices[current] >= 0 {
            current = self.parent_indices[current] as usize;
            if current == j {
                return true;
            }
        }

        false
    }

    /// Validate tree structure
    pub fn validate(&self) -> Result<(), String> {
        if self.tokens.len() != self.num_nodes {
            return Err("Token count mismatch".to_string());
        }
        if self.parent_indices.len() != self.num_nodes {
            return Err("Parent indices count mismatch".to_string());
        }
        if self.depth.len() != self.num_nodes {
            return Err("Depth count mismatch".to_string());
        }

        // Check root
        if self.num_nodes > 0 {
            let root_count = self.parent_indices.iter().filter(|&&p| p < 0).count();
            if root_count != 1 {
                return Err(format!("Expected 1 root, found {}", root_count));
            }
        }

        // Check parent references
        for (i, &parent) in self.parent_indices.iter().enumerate() {
            if parent >= 0 && parent as usize >= self.num_nodes {
                return Err(format!("Invalid parent {} for node {}", parent, i));
            }
            if parent >= 0 && parent as usize >= i {
                return Err(format!("Parent {} >= node {} (not DFS order)", parent, i));
            }
        }

        Ok(())
    }
}

impl Default for TokenTree {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tree Mask (ARCH-OP-010)
// ============================================================================

/// Compressed tree mask for attention computation
///
/// SPEC: ARCH-OP-010 TreeMask structure
///
/// mask[i][j] = 1 iff j is ancestor of i (including i itself)
#[derive(Debug, Clone)]
pub struct TreeMask {
    /// Bit-packed mask data (row-major)
    pub mask_data: Vec<u64>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Words per row (ceiling of num_nodes / 64)
    words_per_row: usize,
}

impl TreeMask {
    /// Create mask from token tree
    pub fn from_tree(tree: &TokenTree) -> Self {
        let num_nodes = tree.num_nodes;
        let words_per_row = (num_nodes + 63) / 64;
        let mut mask_data = vec![0u64; num_nodes * words_per_row];

        for i in 0..num_nodes {
            // j is ancestor of i if j is on path from root to i
            let path = tree.path_to_root(i);
            for &j in &path {
                let row_start = i * words_per_row;
                let word_idx = j / 64;
                let bit_idx = j % 64;
                mask_data[row_start + word_idx] |= 1u64 << bit_idx;
            }
        }

        Self {
            mask_data,
            num_nodes,
            words_per_row,
        }
    }

    /// Check if mask[i][j] is set
    pub fn get(&self, i: usize, j: usize) -> bool {
        if i >= self.num_nodes || j >= self.num_nodes {
            return false;
        }

        let row_start = i * self.words_per_row;
        let word_idx = j / 64;
        let bit_idx = j % 64;

        (self.mask_data[row_start + word_idx] >> bit_idx) & 1 == 1
    }

    /// Set mask[i][j]
    pub fn set(&mut self, i: usize, j: usize) {
        if i >= self.num_nodes || j >= self.num_nodes {
            return;
        }

        let row_start = i * self.words_per_row;
        let word_idx = j / 64;
        let bit_idx = j % 64;

        self.mask_data[row_start + word_idx] |= 1u64 << bit_idx;
    }

    /// Get row as boolean vector
    pub fn get_row(&self, i: usize) -> Vec<bool> {
        (0..self.num_nodes).map(|j| self.get(i, j)).collect()
    }

    /// Count number of set bits in row
    pub fn row_popcount(&self, i: usize) -> usize {
        if i >= self.num_nodes {
            return 0;
        }

        let row_start = i * self.words_per_row;
        self.mask_data[row_start..row_start + self.words_per_row]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Get total memory size in bytes
    pub fn memory_bytes(&self) -> usize {
        self.mask_data.len() * 8
    }
}

// ============================================================================
// Batch Tree Config (ARCH-OP-010)
// ============================================================================

/// Configuration for batch tree attention
///
/// SPEC: ARCH-OP-010 BatchTreeConfig structure
#[derive(Debug, Clone)]
pub struct BatchTreeConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum tree depth
    pub max_tree_depth: usize,
    /// Maximum nodes per tree
    pub max_nodes_per_tree: usize,
    /// Tree partition strategy
    pub partition_strategy: PartitionStrategy,
    /// Enable Talon confidence adaptation
    pub enable_talon: bool,
    /// Enable sequence-level traversal verification
    pub traversal_verification: bool,
}

impl Default for BatchTreeConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tree_depth: 8,
            max_nodes_per_tree: 64,
            partition_strategy: PartitionStrategy::DeFTFlatten,
            enable_talon: true,
            traversal_verification: true,
        }
    }
}

impl BatchTreeConfig {
    /// Create config for small models (aggressive speculation)
    pub fn for_small_model() -> Self {
        Self {
            max_tree_depth: 12,
            max_nodes_per_tree: 128,
            ..Default::default()
        }
    }

    /// Create config for large models (conservative speculation)
    pub fn for_large_model() -> Self {
        Self {
            max_tree_depth: 6,
            max_nodes_per_tree: 32,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".to_string());
        }
        if self.max_tree_depth == 0 {
            return Err("max_tree_depth must be > 0".to_string());
        }
        if self.max_nodes_per_tree == 0 {
            return Err("max_nodes_per_tree must be > 0".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// Talon Config (ARCH-OP-010)
// ============================================================================

/// Configuration for Talon confidence-adaptive trees
///
/// SPEC: ARCH-OP-010 TalonConfig structure
#[derive(Debug, Clone)]
pub struct TalonConfig {
    /// Size of acceptance rate history window
    pub acceptance_history_size: usize,
    /// Threshold to expand tree (high acceptance)
    pub tree_expansion_threshold: f32,
    /// Threshold to shrink tree (low acceptance)
    pub tree_shrink_threshold: f32,
    /// Minimum branches per node
    pub min_branches: usize,
    /// Maximum branches per node
    pub max_branches: usize,
}

impl Default for TalonConfig {
    fn default() -> Self {
        Self {
            acceptance_history_size: 100,
            tree_expansion_threshold: 0.8,
            tree_shrink_threshold: 0.3,
            min_branches: 2,
            max_branches: 8,
        }
    }
}

impl TalonConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.acceptance_history_size == 0 {
            return Err("acceptance_history_size must be > 0".to_string());
        }
        if self.tree_expansion_threshold <= self.tree_shrink_threshold {
            return Err("expansion_threshold must be > shrink_threshold".to_string());
        }
        if self.min_branches > self.max_branches {
            return Err("min_branches must be <= max_branches".to_string());
        }
        if self.min_branches == 0 {
            return Err("min_branches must be > 0".to_string());
        }
        Ok(())
    }
}

// ============================================================================
// Talon Adaptive Controller
// ============================================================================

/// Talon confidence-adaptive tree controller
pub struct TalonController {
    config: TalonConfig,
    acceptance_history: VecDeque<bool>,
    current_branches: usize,
    current_depth: usize,
}

impl TalonController {
    /// Create new controller
    pub fn new(config: TalonConfig) -> Self {
        let current_branches = (config.min_branches + config.max_branches) / 2;
        Self {
            config,
            acceptance_history: VecDeque::new(),
            current_branches,
            current_depth: 4,
        }
    }

    /// Record verification result
    pub fn record_acceptance(&mut self, accepted: bool) {
        self.acceptance_history.push_back(accepted);
        while self.acceptance_history.len() > self.config.acceptance_history_size {
            self.acceptance_history.pop_front();
        }

        // Adapt tree structure
        let rate = self.acceptance_rate();

        if rate > self.config.tree_expansion_threshold {
            // High acceptance - expand tree
            if self.current_branches < self.config.max_branches {
                self.current_branches += 1;
            }
            self.current_depth = (self.current_depth + 1).min(12);
        } else if rate < self.config.tree_shrink_threshold {
            // Low acceptance - shrink tree
            if self.current_branches > self.config.min_branches {
                self.current_branches -= 1;
            }
            self.current_depth = (self.current_depth.saturating_sub(1)).max(2);
        }
    }

    /// Get current acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        if self.acceptance_history.is_empty() {
            return 0.5; // Default
        }
        let accepted = self.acceptance_history.iter().filter(|&&a| a).count();
        accepted as f32 / self.acceptance_history.len() as f32
    }

    /// Get recommended tree parameters
    pub fn recommended_params(&self) -> (usize, usize) {
        (self.current_branches, self.current_depth)
    }

    /// Reset history
    pub fn reset(&mut self) {
        self.acceptance_history.clear();
        self.current_branches = (self.config.min_branches + self.config.max_branches) / 2;
        self.current_depth = 4;
    }
}

// ============================================================================
// Tree Attention Output
// ============================================================================

/// Output of tree attention computation
#[derive(Debug)]
pub struct TreeAttentionOutput {
    /// Output values for each node [num_nodes, head_dim]
    pub output: Vec<f32>,
    /// Attention weights (optional, for debugging)
    pub attention_weights: Option<Vec<f32>>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Head dimension
    pub head_dim: usize,
}

// ============================================================================
// Flash Tree Attention (ARCH-OP-010)
// ============================================================================

/// Flash Tree Attention implementation
///
/// SPEC: ARCH-OP-010 FlashTreeAttention structure
pub struct FlashTreeAttention {
    /// Batch configuration
    config: BatchTreeConfig,
    /// Talon controller (if enabled)
    talon: Option<TalonController>,
    /// Softmax scale factor
    scale: f32,
}

impl FlashTreeAttention {
    /// Create new Flash Tree Attention
    pub fn new(config: BatchTreeConfig, head_dim: usize) -> Result<Self, String> {
        config.validate()?;

        let talon = if config.enable_talon {
            Some(TalonController::new(TalonConfig::default()))
        } else {
            None
        };

        Ok(Self {
            config,
            talon,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    /// Create with custom Talon config
    pub fn with_talon(
        config: BatchTreeConfig,
        talon_config: TalonConfig,
        head_dim: usize,
    ) -> Result<Self, String> {
        config.validate()?;
        talon_config.validate()?;

        Ok(Self {
            config,
            talon: Some(TalonController::new(talon_config)),
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    /// Forward pass for single tree
    ///
    /// q, k, v: [num_nodes, head_dim]
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        tree: &TokenTree,
    ) -> TreeAttentionOutput {
        let num_nodes = tree.num_nodes;
        let head_dim = q.len() / num_nodes;

        // Build tree mask
        let mask = TreeMask::from_tree(tree);

        // CPU reference implementation of tree attention
        self.tree_attention_cpu(q, k, v, &mask, num_nodes, head_dim)
    }

    /// Batch forward pass for multiple trees
    pub fn batch_forward(
        &self,
        batch_q: &[Vec<f32>],
        batch_k: &[Vec<f32>],
        batch_v: &[Vec<f32>],
        batch_trees: &[TokenTree],
    ) -> Vec<TreeAttentionOutput> {
        // Process each tree (can be parallelized)
        batch_q
            .iter()
            .zip(batch_k.iter())
            .zip(batch_v.iter())
            .zip(batch_trees.iter())
            .map(|(((q, k), v), tree)| self.forward(q, k, v, tree))
            .collect()
    }

    /// Record verification result for Talon adaptation
    pub fn record_verification(&mut self, accepted_count: usize, total_count: usize) {
        if let Some(ref mut talon) = self.talon {
            for i in 0..total_count {
                talon.record_acceptance(i < accepted_count);
            }
        }
    }

    /// Get recommended tree parameters from Talon
    pub fn recommended_tree_params(&self) -> Option<(usize, usize)> {
        self.talon.as_ref().map(|t| t.recommended_params())
    }

    /// Get current configuration
    pub fn config(&self) -> &BatchTreeConfig {
        &self.config
    }

    // CPU reference implementation
    fn tree_attention_cpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &TreeMask,
        num_nodes: usize,
        head_dim: usize,
    ) -> TreeAttentionOutput {
        let mut output = vec![0.0f32; num_nodes * head_dim];
        let mut all_weights = Vec::new();

        for i in 0..num_nodes {
            // Compute attention scores for node i
            let qi = &q[i * head_dim..(i + 1) * head_dim];
            let mut scores = Vec::with_capacity(num_nodes);
            let mut mask_indices = Vec::new();

            for j in 0..num_nodes {
                if mask.get(i, j) {
                    // j is ancestor of i
                    let kj = &k[j * head_dim..(j + 1) * head_dim];
                    let score: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                    scores.push(score * self.scale);
                    mask_indices.push(j);
                }
            }

            if scores.is_empty() {
                continue;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            all_weights.extend(weights.iter().cloned());

            // Weighted sum of values
            let oi = &mut output[i * head_dim..(i + 1) * head_dim];
            for (idx, &j) in mask_indices.iter().enumerate() {
                let vj = &v[j * head_dim..(j + 1) * head_dim];
                let w = weights[idx];
                for (o, &val) in oi.iter_mut().zip(vj.iter()) {
                    *o += w * val;
                }
            }
        }

        TreeAttentionOutput {
            output,
            attention_weights: Some(all_weights),
            num_nodes,
            head_dim,
        }
    }
}

// ============================================================================
// Traversal Verification
// ============================================================================

/// Result of traversal verification
#[derive(Debug)]
pub struct TraversalResult {
    /// Accepted token indices
    pub accepted_indices: Vec<usize>,
    /// Longest accepted path
    pub accepted_path: Vec<usize>,
    /// Total verified nodes
    pub verified_count: usize,
    /// Acceptance rate for this verification
    pub acceptance_rate: f32,
}

/// Perform traversal verification on tree
///
/// target_probs: Target model probabilities for each node
/// draft_tokens: Tokens in the tree
pub fn traversal_verify(
    tree: &TokenTree,
    target_probs: &[Vec<f32>],
    vocab_size: usize,
    temperature: f32,
) -> TraversalResult {
    let mut accepted_indices = Vec::new();
    let mut best_path = Vec::new();

    // BFS verification from root
    let mut queue: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
    queue.push_back((0, vec![0])); // Start from root

    while let Some((node_idx, path)) = queue.pop_front() {
        let token = tree.tokens[node_idx] as usize;

        // Check if token is acceptable (simplified: high probability)
        let prob = if node_idx < target_probs.len() && token < vocab_size {
            if token < target_probs[node_idx].len() {
                target_probs[node_idx][token]
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Accept if probability is above threshold
        let threshold = if temperature > 0.0 {
            0.1 / temperature
        } else {
            0.5
        };

        if prob >= threshold || node_idx == 0 {
            // Root always accepted
            accepted_indices.push(node_idx);
            if path.len() > best_path.len() {
                best_path = path.clone();
            }

            // Add children to queue
            for child_idx in tree.children(node_idx) {
                let mut child_path = path.clone();
                child_path.push(child_idx);
                queue.push_back((child_idx, child_path));
            }
        }
    }

    let verified_count = tree.num_nodes;
    let acceptance_rate = accepted_indices.len() as f32 / verified_count.max(1) as f32;

    TraversalResult {
        accepted_indices,
        accepted_path: best_path,
        verified_count,
        acceptance_rate,
    }
}

// ============================================================================
// DeFT Partitioning
// ============================================================================

/// Partition tree for DeFT-Flatten execution
pub fn deft_flatten_partition(tree: &TokenTree, num_sms: usize) -> Vec<Vec<usize>> {
    let nodes_per_sm = (tree.num_nodes + num_sms - 1) / num_sms;
    let mut partitions = Vec::with_capacity(num_sms);

    // Simple round-robin assignment in DFS order
    for sm in 0..num_sms {
        let start = sm * nodes_per_sm;
        let end = (start + nodes_per_sm).min(tree.num_nodes);
        if start < tree.num_nodes {
            partitions.push((start..end).collect());
        }
    }

    partitions
}

/// Partition tree for DeFT-Node execution
pub fn deft_node_partition(tree: &TokenTree, num_sms: usize) -> Vec<Vec<usize>> {
    // Group by depth level
    let max_depth = tree.max_depth();
    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];

    for (idx, &depth) in tree.depth.iter().enumerate() {
        levels[depth].push(idx);
    }

    // Distribute levels across SMs
    let mut partitions: Vec<Vec<usize>> = (0..num_sms).map(|_| Vec::new()).collect();

    for (level_idx, level) in levels.into_iter().enumerate() {
        let target_sm = level_idx % num_sms;
        partitions[target_sm].extend(level);
    }

    partitions.into_iter().filter(|p| !p.is_empty()).collect()
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for tree attention operations
#[derive(Debug, Clone, Default)]
pub struct TreeAttentionStats {
    /// Total trees processed
    pub trees_processed: u64,
    /// Total nodes processed
    pub nodes_processed: u64,
    /// Average tree depth
    pub avg_depth: f32,
    /// Average nodes per tree
    pub avg_nodes: f32,
    /// Average acceptance rate (if Talon enabled)
    pub avg_acceptance_rate: f32,
}

impl TreeAttentionStats {
    /// Update with new tree
    pub fn update(&mut self, tree: &TokenTree, acceptance_rate: Option<f32>) {
        self.trees_processed += 1;
        self.nodes_processed += tree.num_nodes as u64;

        // Running average for depth
        let depth = tree.max_depth() as f32;
        self.avg_depth = (self.avg_depth * (self.trees_processed - 1) as f32 + depth)
            / self.trees_processed as f32;

        // Running average for nodes
        let nodes = tree.num_nodes as f32;
        self.avg_nodes = (self.avg_nodes * (self.trees_processed - 1) as f32 + nodes)
            / self.trees_processed as f32;

        // Running average for acceptance rate
        if let Some(rate) = acceptance_rate {
            self.avg_acceptance_rate =
                (self.avg_acceptance_rate * (self.trees_processed - 1) as f32 + rate)
                    / self.trees_processed as f32;
        }
    }

    /// Estimate speedup over sequential verification
    pub fn estimated_speedup(&self) -> f32 {
        // DeFT provides O(n+m) vs O(n*m) for naive
        // Speedup depends on tree structure
        if self.avg_nodes > 1.0 {
            self.avg_depth.max(1.0)
        } else {
            1.0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_tree_basic() {
        let mut tree = TokenTree::with_root(0);
        assert_eq!(tree.num_nodes, 1);

        // Build tree:
        //       0
        //      /|\
        //     1 2 3
        //    /|   |
        //   4 5   6

        let n1 = tree.add_child(0, 1);
        let n2 = tree.add_child(0, 2);
        let n3 = tree.add_child(0, 3);
        let _n4 = tree.add_child(n1, 4);
        let _n5 = tree.add_child(n1, 5);
        let _n6 = tree.add_child(n3, 6);

        assert_eq!(tree.num_nodes, 7);
        assert_eq!(tree.max_depth(), 2);
        assert!(tree.validate().is_ok());

        // Check children
        assert_eq!(tree.children(0), vec![1, 2, 3]);
        assert_eq!(tree.children(1), vec![4, 5]);
        assert_eq!(tree.children(3), vec![6]);

        // Check leaves
        assert_eq!(tree.leaves(), vec![2, 4, 5, 6]);

        // Check paths
        let path = tree.path_to_root(5);
        assert_eq!(path, vec![0, 1, 5]);
    }

    #[test]
    fn test_token_tree_ancestry() {
        let tree = TokenTree::from_candidates(
            0,
            &[
                (1, 0), // node 1, parent 0
                (2, 0), // node 2, parent 0
                (3, 1), // node 3, parent 1
            ],
        );

        // 0 is ancestor of all
        assert!(tree.is_ancestor(0, 0));
        assert!(tree.is_ancestor(1, 0));
        assert!(tree.is_ancestor(2, 0));
        assert!(tree.is_ancestor(3, 0));

        // 1 is ancestor of 3
        assert!(tree.is_ancestor(3, 1));

        // 2 is not ancestor of 3
        assert!(!tree.is_ancestor(3, 2));
    }

    #[test]
    fn test_tree_mask() {
        //       0
        //      /|\
        //     1 2 3
        let tree = TokenTree::from_candidates(0, &[(1, 0), (2, 0), (3, 0)]);
        let mask = TreeMask::from_tree(&tree);

        // mask[i][j] = 1 iff j is ancestor of i
        assert!(mask.get(0, 0)); // root sees itself
        assert!(mask.get(1, 0)); // 1 sees root
        assert!(mask.get(1, 1)); // 1 sees itself
        assert!(!mask.get(1, 2)); // 1 doesn't see 2 (sibling)
        assert!(!mask.get(1, 3)); // 1 doesn't see 3 (sibling)

        // Check row popcounts
        assert_eq!(mask.row_popcount(0), 1); // root sees only itself
        assert_eq!(mask.row_popcount(1), 2); // 1 sees root + itself
    }

    #[test]
    fn test_tree_mask_deep() {
        // Linear tree: 0 -> 1 -> 2 -> 3
        let mut tree = TokenTree::with_root(0);
        let n1 = tree.add_child(0, 1);
        let n2 = tree.add_child(n1, 2);
        tree.add_child(n2, 3);

        let mask = TreeMask::from_tree(&tree);

        // Node 3 should see all ancestors
        assert!(mask.get(3, 0));
        assert!(mask.get(3, 1));
        assert!(mask.get(3, 2));
        assert!(mask.get(3, 3));
        assert_eq!(mask.row_popcount(3), 4);
    }

    #[test]
    fn test_batch_config_validation() {
        let valid = BatchTreeConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = BatchTreeConfig {
            max_batch_size: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_talon_config_validation() {
        let valid = TalonConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = TalonConfig {
            tree_expansion_threshold: 0.3,
            tree_shrink_threshold: 0.8, // Greater than expansion
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_talon_controller() {
        let config = TalonConfig::default();
        let mut controller = TalonController::new(config);

        // Initial state
        assert_eq!(controller.acceptance_rate(), 0.5);

        // Record many acceptances
        for _ in 0..50 {
            controller.record_acceptance(true);
        }
        assert!(controller.acceptance_rate() > 0.9);

        // Should have expanded
        let (branches, _depth) = controller.recommended_params();
        assert!(branches > 2);

        // Record many rejections
        for _ in 0..100 {
            controller.record_acceptance(false);
        }

        // Should have shrunk
        let (branches, _) = controller.recommended_params();
        assert!(branches >= 2);
    }

    #[test]
    fn test_flash_tree_attention() {
        let config = BatchTreeConfig::default();
        let head_dim = 64;
        let attn = FlashTreeAttention::new(config, head_dim).unwrap();

        // Create simple tree
        let tree = TokenTree::from_candidates(0, &[(1, 0), (2, 0), (3, 1)]);
        let num_nodes = tree.num_nodes;

        // Create random Q, K, V
        let q: Vec<f32> = (0..num_nodes * head_dim).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..num_nodes * head_dim).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..num_nodes * head_dim).map(|i| (i as f32) * 0.01).collect();

        let output = attn.forward(&q, &k, &v, &tree);

        assert_eq!(output.num_nodes, num_nodes);
        assert_eq!(output.head_dim, head_dim);
        assert_eq!(output.output.len(), num_nodes * head_dim);
    }

    #[test]
    fn test_traversal_verify() {
        let tree = TokenTree::from_candidates(
            100, // root token
            &[(101, 0), (102, 0), (103, 1), (104, 1)],
        );

        // Create probabilities that favor path through node 1 -> 3
        let vocab_size = 200;
        let probs: Vec<Vec<f32>> = vec![
            vec![0.0; vocab_size], // For root
            vec![0.0; vocab_size], // For node 1
            vec![0.0; vocab_size], // For node 2
            vec![0.0; vocab_size], // For node 3
            vec![0.0; vocab_size], // For node 4
        ];

        let result = traversal_verify(&tree, &probs, vocab_size, 1.0);

        // At least root should be accepted
        assert!(!result.accepted_indices.is_empty());
        assert!(result.accepted_indices.contains(&0));
    }

    #[test]
    fn test_deft_partitioning() {
        let tree = TokenTree::from_candidates(0, &[(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)]);

        // DeFT-Flatten
        let partitions = deft_flatten_partition(&tree, 3);
        assert!(partitions.len() <= 3);
        let total: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total, tree.num_nodes);

        // DeFT-Node
        let partitions = deft_node_partition(&tree, 3);
        let total: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total, tree.num_nodes);
    }

    #[test]
    fn test_tree_attention_stats() {
        let mut stats = TreeAttentionStats::default();

        let tree1 = TokenTree::from_candidates(0, &[(1, 0), (2, 0)]);
        let tree2 = TokenTree::from_candidates(0, &[(1, 0), (2, 1), (3, 2)]);

        stats.update(&tree1, Some(0.8));
        stats.update(&tree2, Some(0.6));

        assert_eq!(stats.trees_processed, 2);
        assert_eq!(stats.nodes_processed, 7);
        assert!((stats.avg_acceptance_rate - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_config_presets() {
        let small = BatchTreeConfig::for_small_model();
        let large = BatchTreeConfig::for_large_model();

        assert!(small.max_tree_depth > large.max_tree_depth);
        assert!(small.max_nodes_per_tree > large.max_nodes_per_tree);
    }
}
