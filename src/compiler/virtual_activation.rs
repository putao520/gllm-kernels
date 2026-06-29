//! Phase D.2: 虚拟激活流 — 跨层 activation 身份映射 (SSOT: §3.6, §9.3)
//!
//! 核心洞察: 层 N 的 output activation 就是层 N+1 的 input activation。
//! 当前每层分配独立的 input/output buffer → ceil(N/2) 个 buffer。
//! 正确做法: 2 个 buffer (ping/pong) 交替使用 → 总 activation 大小 = 2 × hidden_dim。
//!
//! ActivationSwap(ptr_a, ptr_b): 层循环末尾交换 ptr，零数据拷贝。

use std::collections::HashMap;
use crate::compiler::graph::{CompilerGraph, TensorId};
use crate::compiler::fusion::FusionPlan;

/// 激活 buffer 角色标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationRole {
    /// 当前层的输入 (由上一层写入或初始 input)
    Input,
    /// 当前层的输出 (写入后成为下一层的输入)
    Output,
}

/// 虚拟激活映射: 层间 activation 的身份映射
#[derive(Debug, Clone)]
pub struct VirtualActivationMap {
    /// 每个层间 activation tensor 的分配角色
    /// TensorId → (buffer_index, role)
    /// buffer_index 只有 0 (ping) 和 1 (pong)
    pub activation_assignments: HashMap<TensorId, ActivationSlot>,
    /// 每层循环的 ptr 交换序列
    pub swap_sequence: Vec<ActivationSwap>,
    /// 总 activation buffer 数 (始终为 2)
    pub num_buffers: usize,
    /// 单个 buffer 的大小 (hidden_dim × elem_bytes)
    pub buffer_size_bytes: usize,
    /// 相比独立分配节省的字节数
    pub bytes_saved: usize,
}

/// 激活 slot 分配
#[derive(Debug, Clone)]
pub struct ActivationSlot {
    /// buffer 索引 (0=ping, 1=pong)
    pub buffer_idx: usize,
    /// 角色
    pub role: ActivationRole,
    /// 字节偏移 (在 buffer 内)
    pub byte_offset: usize,
}

/// 层间 ptr 交换描述 (SPEC §3.7 ActivationSwap)
#[derive(Debug, Clone)]
pub struct ActivationSwap {
    /// 层索引
    pub layer_idx: usize,
    /// 交换前的 ping buffer ptr
    pub ping_ptr: TensorId,
    /// 交换前的 pong buffer ptr
    pub pong_ptr: TensorId,
}

impl VirtualActivationMap {
    /// 分析激活流并生成虚拟激活映射
    ///
    /// 识别层间 activation tensor，将它们映射到 2 个 ping-pong buffer。
    pub fn analyze(
        graph: &CompilerGraph,
        plan: &FusionPlan,
    ) -> Self {
        eprintln!("[VAM] analyze: layer_loop_config={:?} activation_alias={:?}",
            graph.layer_loop_config.as_ref().map(|c| (c.num_layers, c.weight_stride)),
            graph.layer_loop_config.as_ref().and_then(|c| c.activation_alias));
        let mut activation_assignments = HashMap::new();
        let mut swap_sequence = Vec::new();

        // Step 1: 识别所有层间 activation tensor
        // 层 N 的 output[0] = 层 N+1 的 input[0] (通常)
        let layer_activations = find_layer_activations(graph, plan);

        if layer_activations.is_empty() {
            return VirtualActivationMap {
                activation_assignments,
                swap_sequence,
                num_buffers: 0,
                buffer_size_bytes: 0,
                bytes_saved: 0,
            };
        }

        // Step 2: 计算 buffer 大小
        // Must use tensor_numel_for_alloc (respects max_seq_len for Symbolic dims)
        // instead of concrete_bytes (which treats Symbolic as 1 element).
        let max_activation_bytes = layer_activations.iter()
            .map(|(tid, _)| {
                graph.tensor_numel_for_alloc(*tid, graph.max_seq_len)
                    .map(|numel| numel * 4) // F32 intermediate
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);

        // Step 3: 分配 ping-pong slots
        // 偶数层: input → ping(0), output → pong(1)
        // 奇数层: input → pong(1), output → ping(0)
        for (idx, (tid, layer_idx)) in layer_activations.iter().enumerate() {
            let buffer_idx = idx % 2;
            let role = if idx % 2 == 0 { ActivationRole::Input } else { ActivationRole::Output };
            activation_assignments.insert(*tid, ActivationSlot {
                buffer_idx,
                role,
                byte_offset: 0,
            });

            // 每层结束生成 swap
            if idx > 0 && idx % 2 == 0 {
                // 偶数 index = 新层开始 = 需要交换
                swap_sequence.push(ActivationSwap {
                    layer_idx: *layer_idx,
                    ping_ptr: *tid,
                    pong_ptr: layer_activations[idx - 1].0,
                });
            }
        }

        // Step 4: 计算节省
        let total_original = layer_activations.len() * max_activation_bytes;
        let total_virtual = 2 * max_activation_bytes;
        let bytes_saved = total_original.saturating_sub(total_virtual);

        VirtualActivationMap {
            activation_assignments,
            swap_sequence,
            num_buffers: 2,
            buffer_size_bytes: max_activation_bytes,
            bytes_saved,
        }
    }
}

/// 找到所有层间 activation tensor — 仅返回 `layer_loop_config.activation_alias` 中的两个 tensor。
///
/// activation_alias = (layer_input, layer_output):
/// - layer_input: 层循环的输入 tensor (embedding → 第一层 → 第二层 → ...)
/// - layer_output: 层循环的输出 tensor (最后一层的残差输出)
///
/// 这两个 tensor 共享 ping-pong buffer，层循环每轮迭代后 ActivationSwap 交换指针。
/// 其他 "layer." 组的中间 tensor (q_proj, k_proj, attn_out 等) 是层内临时变量，
/// 需要通过正常 lifetime coloring 获取独立 buffer slot，不能放入 ping-pong buffer。
fn find_layer_activations(
    graph: &CompilerGraph,
    _plan: &FusionPlan,
) -> Vec<(TensorId, usize)> {
    use std::collections::HashSet;
    // BCE-20260629-005: 排除 Gather/QuantGather 输出（如 embedding）。
    // Gather 输出是数据加载（写入），不是层间激活交换（in-place 读写）。
    // 如果 Gather 输出被当作 activation tensor，VAM 会把它分配到 ping/pong buffer，
    // 而 resolver materialize 会返回 activation_ping_ptr → gather 写到 ping buffer
    // 而非 scratchpad → DIAG 读 scratchpad offset 0 读不到 → NaN。
    let gather_output_tids: HashSet<TensorId> = graph.ops.iter()
        .filter_map(|op| match &op.op {
            crate::compiler::graph::Op::Gather { .. } | crate::compiler::graph::Op::QuantGather { .. } => op.outputs.first().copied(),
            _ => None,
        })
        .collect();
    if let Some(ref cfg) = graph.layer_loop_config {
        if let Some((input_tid, output_tid)) = cfg.activation_alias {
            // 排除 Gather 输出，返回不含 Gather 输出的 activation tensors
            let result: Vec<(TensorId, usize)> = [(input_tid, 0), (output_tid, 0)]
                .into_iter()
                .filter(|(tid, _)| !gather_output_tids.contains(tid))
                .collect();
            return result;
        }
    }
    if let Some(ref cfg) = graph.hetero_layer_loop_config {
        if let Some(&(input_tid, output_tid)) = cfg.activation_aliases.first() {
            let result: Vec<(TensorId, usize)> = [(input_tid, 0), (output_tid, 0)]
                .into_iter()
                .filter(|(tid, _)| !gather_output_tids.contains(tid))
                .collect();
            return result;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = CompilerGraph::new();
        let plan = FusionPlan {
            groups: Vec::new(),
            op_to_group: HashMap::new(),
        };
        let vam = VirtualActivationMap::analyze(&graph, &plan);
        assert_eq!(vam.num_buffers, 0);
        assert_eq!(vam.bytes_saved, 0);
    }

    #[test]
    fn test_activation_role_equality() {
        assert_eq!(ActivationRole::Input, ActivationRole::Input);
        assert_ne!(ActivationRole::Input, ActivationRole::Output);
    }

    #[test]
    fn test_activation_slot_buffer_idx() {
        let slot = ActivationSlot {
            buffer_idx: 0,
            role: ActivationRole::Input,
            byte_offset: 0,
        };
        assert_eq!(slot.buffer_idx, 0);
        assert!(matches!(slot.role, ActivationRole::Input));
    }

    // ── Test 4: ActivationRole Copy and Hash ──

    #[test]
    fn activation_role_copy_and_hash() {
        // Arrange
        let input = ActivationRole::Input;
        let copied = input;

        // Act & Assert — Copy
        assert_eq!(input, copied);

        // Hash — both roles should produce consistent hashes
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ActivationRole::Input);
        set.insert(ActivationRole::Output);
        assert_eq!(set.len(), 2, "Input and Output should hash to distinct entries");
        assert!(set.contains(&ActivationRole::Input));
        assert!(set.contains(&ActivationRole::Output));
    }

    // ── Test 5: ActivationRole Debug format ──

    #[test]
    fn activation_role_debug_format() {
        // Arrange & Act
        let debug_input = format!("{:?}", ActivationRole::Input);
        let debug_output = format!("{:?}", ActivationRole::Output);

        // Assert — Debug should contain the variant name
        assert!(debug_input.contains("Input"));
        assert!(debug_output.contains("Output"));
    }

    // ── Test 6: ActivationSlot clone preserves all fields ──

    #[test]
    fn activation_slot_clone_preserves_fields() {
        // Arrange
        let slot = ActivationSlot {
            buffer_idx: 1,
            role: ActivationRole::Output,
            byte_offset: 256,
        };

        // Act
        let cloned = slot.clone();

        // Assert
        assert_eq!(cloned.buffer_idx, 1);
        assert_eq!(cloned.role, ActivationRole::Output);
        assert_eq!(cloned.byte_offset, 256);
    }

    // ── Test 7: ActivationSwap construction and Debug ──

    #[test]
    fn activation_swap_construction_and_debug() {
        // Arrange
        let swap = ActivationSwap {
            layer_idx: 5,
            ping_ptr: TensorId(10),
            pong_ptr: TensorId(20),
        };

        // Act
        let debug = format!("{:?}", swap);
        let cloned = swap.clone();

        // Assert
        assert_eq!(swap.layer_idx, 5);
        assert_eq!(swap.ping_ptr, TensorId(10));
        assert_eq!(swap.pong_ptr, TensorId(20));
        assert_eq!(cloned.layer_idx, 5);
        assert!(debug.contains("5"));
    }

    // ── Test 8: VirtualActivationMap empty has correct defaults ──

    #[test]
    fn virtual_activation_map_empty_defaults() {
        // Arrange — construct manually to test defaults
        let vam = VirtualActivationMap {
            activation_assignments: HashMap::new(),
            swap_sequence: Vec::new(),
            num_buffers: 0,
            buffer_size_bytes: 0,
            bytes_saved: 0,
        };

        // Assert
        assert!(vam.activation_assignments.is_empty());
        assert!(vam.swap_sequence.is_empty());
        assert_eq!(vam.num_buffers, 0);
        assert_eq!(vam.buffer_size_bytes, 0);
        assert_eq!(vam.bytes_saved, 0);
    }

    // ── Test 9: VirtualActivationMap Debug and Clone ──

    #[test]
    fn virtual_activation_map_debug_and_clone() {
        // Arrange
        let vam = VirtualActivationMap {
            activation_assignments: HashMap::new(),
            swap_sequence: Vec::new(),
            num_buffers: 2,
            buffer_size_bytes: 4096,
            bytes_saved: 1024,
        };

        // Act
        let debug = format!("{:?}", vam);
        let cloned = vam.clone();

        // Assert
        assert!(debug.contains("2"), "should contain num_buffers");
        assert_eq!(cloned.num_buffers, 2);
        assert_eq!(cloned.buffer_size_bytes, 4096);
        assert_eq!(cloned.bytes_saved, 1024);
    }

    // ── Test 10: ActivationSlot with non-zero byte_offset ──

    #[test]
    fn activation_slot_pong_buffer_with_offset() {
        // Arrange
        let slot = ActivationSlot {
            buffer_idx: 1,
            role: ActivationRole::Output,
            byte_offset: 512,
        };

        // Assert
        assert_eq!(slot.buffer_idx, 1);
        assert_eq!(slot.role, ActivationRole::Output);
        assert_eq!(slot.byte_offset, 512);
        let debug = format!("{:?}", slot);
        assert!(debug.contains("512"));
    }

    // ── Test 11: VirtualActivationMap analyze with graph lacking layer_loop_config ──

    #[test]
    fn virtual_activation_map_analyze_no_layer_loop_config() {
        // Arrange — CompilerGraph::new() has no layer_loop_config
        let graph = CompilerGraph::new();
        let plan = FusionPlan {
            groups: Vec::new(),
            op_to_group: HashMap::new(),
        };

        // Act
        let vam = VirtualActivationMap::analyze(&graph, &plan);

        // Assert — should return empty result (no activations found)
        assert_eq!(vam.num_buffers, 0);
        assert_eq!(vam.buffer_size_bytes, 0);
        assert_eq!(vam.bytes_saved, 0);
        assert!(vam.activation_assignments.is_empty());
        assert!(vam.swap_sequence.is_empty());
    }

    // ── Test 12: ActivationSwap with layer_idx zero ──

    #[test]
    fn activation_swap_layer_idx_zero() {
        // Arrange
        let swap = ActivationSwap {
            layer_idx: 0,
            ping_ptr: TensorId(0),
            pong_ptr: TensorId(1),
        };

        // Assert
        assert_eq!(swap.layer_idx, 0);
        assert_eq!(swap.ping_ptr, TensorId(0));
        assert_eq!(swap.pong_ptr, TensorId(1));
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn activation_role_all_variants_distinct() {
        // Verify Input and Output are the only variants and they differ
        assert_ne!(ActivationRole::Input, ActivationRole::Output);
        // Verify exhaustive match compiles
        let _ = match ActivationRole::Input {
            ActivationRole::Input => "in",
            ActivationRole::Output => "out",
        };
    }

    #[test]
    fn activation_slot_buffer_idx_pong() {
        // Arrange — pong slot with nonzero offset
        let slot = ActivationSlot {
            buffer_idx: 1,
            role: ActivationRole::Output,
            byte_offset: 1024,
        };
        // Assert
        assert_eq!(slot.buffer_idx, 1);
        assert_eq!(slot.role, ActivationRole::Output);
        assert_eq!(slot.byte_offset, 1024);
    }

    #[test]
    fn activation_swap_same_ptrs_valid() {
        // Edge case: ping and pong can be the same tensor (unusual but legal struct)
        let swap = ActivationSwap {
            layer_idx: 3,
            ping_ptr: TensorId(42),
            pong_ptr: TensorId(42),
        };
        assert_eq!(swap.ping_ptr, swap.pong_ptr);
    }

    #[test]
    fn virtual_activation_map_manual_with_assignments() {
        // Arrange — manually construct a map with one assignment
        let mut assignments = HashMap::new();
        assignments.insert(TensorId(10), ActivationSlot {
            buffer_idx: 0,
            role: ActivationRole::Input,
            byte_offset: 0,
        });
        let vam = VirtualActivationMap {
            activation_assignments: assignments,
            swap_sequence: vec![],
            num_buffers: 2,
            buffer_size_bytes: 8192,
            bytes_saved: 32768,
        };

        // Assert
        assert_eq!(vam.activation_assignments.len(), 1);
        assert!(vam.activation_assignments.contains_key(&TensorId(10)));
        assert_eq!(vam.num_buffers, 2);
        assert_eq!(vam.buffer_size_bytes, 8192);
        assert_eq!(vam.bytes_saved, 32768);
    }

    #[test]
    fn virtual_activation_map_bytes_saved_calculation() {
        // For N layers with hidden_dim activations:
        // original = N * hidden_dim * 4 bytes
        // virtual = 2 * hidden_dim * 4 bytes
        // saved = (N - 2) * hidden_dim * 4
        let hidden_dim = 4096;
        let n_layers = 32;
        let total_original = n_layers * hidden_dim * 4;
        let total_virtual = 2 * hidden_dim * 4;
        let expected_saved = total_original - total_virtual;
        assert_eq!(expected_saved, 30 * hidden_dim * 4);
    }

    #[test]
    fn activation_slot_debug_format() {
        let slot = ActivationSlot {
            buffer_idx: 1,
            role: ActivationRole::Output,
            byte_offset: 2048,
        };
        let debug = format!("{:?}", slot);
        assert!(debug.contains("Output"), "got: {debug}");
        assert!(debug.contains("2048"), "should contain byte_offset");
    }

    #[test]
    fn activation_swap_clone_is_independent() {
        let swap = ActivationSwap {
            layer_idx: 7,
            ping_ptr: TensorId(100),
            pong_ptr: TensorId(200),
        };
        let cloned = swap.clone();
        assert_eq!(cloned.layer_idx, swap.layer_idx);
        assert_eq!(cloned.ping_ptr, swap.ping_ptr);
        assert_eq!(cloned.pong_ptr, swap.pong_ptr);
    }

    #[test]
    fn virtual_activation_map_clone_preserves_everything() {
        let vam = VirtualActivationMap {
            activation_assignments: HashMap::new(),
            swap_sequence: vec![ActivationSwap {
                layer_idx: 1,
                ping_ptr: TensorId(5),
                pong_ptr: TensorId(6),
            }],
            num_buffers: 2,
            buffer_size_bytes: 4096,
            bytes_saved: 8192,
        };
        let cloned = vam.clone();
        assert_eq!(cloned.swap_sequence.len(), 1);
        assert_eq!(cloned.num_buffers, 2);
        assert_eq!(cloned.buffer_size_bytes, 4096);
        assert_eq!(cloned.bytes_saved, 8192);
    }

    #[test]
    fn activation_role_in_hashmap() {
        let mut map = HashMap::new();
        map.insert(ActivationRole::Input, "layer_input");
        map.insert(ActivationRole::Output, "layer_output");
        assert_eq!(map.get(&ActivationRole::Input), Some(&"layer_input"));
        assert_eq!(map.get(&ActivationRole::Output), Some(&"layer_output"));
    }

    #[test]
    fn virtual_activation_map_analyze_empty_plan() {
        // Arrange — plan with no groups, graph with no config
        let graph = CompilerGraph::new();
        let plan = FusionPlan {
            groups: vec![],
            op_to_group: HashMap::new(),
        };
        // Act
        let vam = VirtualActivationMap::analyze(&graph, &plan);
        // Assert — should gracefully return empty
        assert!(vam.swap_sequence.is_empty());
    }
}
