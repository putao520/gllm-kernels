//! TEST-JIT-GRAPH-001~002: SymDim / ShapeBinding 单元测试
//!
//! TEST-JIT-GRAPH-001: SymDim::Symbolic 图编译一次，不同 total_seq binding 解析正确
//! TEST-JIT-GRAPH-002: ShapeBinding API 完整性验证

use gllm_kernels::compiler::{ShapeBinding, SymDim};

/// TEST-JIT-GRAPH-001: Concrete 维度直接返回值，Symbolic 维度从 binding 解析
#[test]
fn test_jit_graph_001_sym_dim_resolve() {
    // Concrete 维度：编译时已知，不依赖 binding
    let concrete = SymDim::Concrete(128);
    let empty_binding = ShapeBinding::new();
    assert_eq!(concrete.resolve(&empty_binding).unwrap(), 128);

    // Symbolic 维度：运行时从 binding 解析
    let sym = SymDim::Symbolic("total_seq".to_string());

    // binding total_seq=10
    let b10 = ShapeBinding::new().bind("total_seq", 10);
    assert_eq!(sym.resolve(&b10).unwrap(), 10);

    // 同一 SymDim，不同 binding → 不同值（模拟 decode 步进）
    let b20 = ShapeBinding::new().bind("total_seq", 20);
    assert_eq!(sym.resolve(&b20).unwrap(), 20);

    let b100 = ShapeBinding::new().bind("total_seq", 100);
    assert_eq!(sym.resolve(&b100).unwrap(), 100);

    // 未绑定时返回 Err，不 panic
    let result = sym.resolve(&empty_binding);
    assert!(result.is_err(), "unresolved symbolic dim must return Err");
    assert!(result.unwrap_err().contains("total_seq"));
}

/// TEST-JIT-GRAPH-001 补充：as_concrete() 行为
#[test]
fn test_jit_graph_001_as_concrete() {
    assert_eq!(SymDim::Concrete(42).as_concrete(), Some(42));
    assert_eq!(SymDim::Symbolic("x".into()).as_concrete(), None);
}

/// TEST-JIT-GRAPH-001 补充：From<usize> 转换
#[test]
fn test_jit_graph_001_from_usize() {
    let dim: SymDim = 64usize.into();
    assert_eq!(dim, SymDim::Concrete(64));
}

/// TEST-JIT-GRAPH-002: ShapeBinding API — bind/insert/get/resolve
#[test]
fn test_jit_graph_002_shape_binding_api() {
    // builder 风格
    let binding = ShapeBinding::new()
        .bind("total_seq", 42)
        .bind("batch_size", 1)
        .bind("hidden", 768);

    assert_eq!(binding.get("total_seq"), Some(&42));
    assert_eq!(binding.get("batch_size"), Some(&1));
    assert_eq!(binding.get("hidden"), Some(&768));
    assert_eq!(binding.get("nonexistent"), None);

    // resolve via binding.resolve()
    let sym = SymDim::Symbolic("total_seq".into());
    assert_eq!(binding.resolve(&sym).unwrap(), 42);

    // insert 风格
    let mut b2 = ShapeBinding::new();
    b2.insert("seq_len", 32);
    assert_eq!(b2.get("seq_len"), Some(&32));

    // From array
    let b3 = ShapeBinding::from([("a", 1usize), ("b", 2usize)]);
    assert_eq!(b3.get("a"), Some(&1));
    assert_eq!(b3.get("b"), Some(&2));
}

/// TEST-JIT-GRAPH-002 补充：多步 decode 模拟 — 同一 SymDim 每步 binding 不同
/// 验证"编译一次，不同 total_seq 复用"的语义正确性
#[test]
fn test_jit_graph_002_decode_step_simulation() {
    let total_seq_dim = SymDim::Symbolic("total_seq".into());

    // 模拟 prefill + 10 decode 步
    for step in 0..=10usize {
        let total_seq = 16 + step; // prefill=16, 然后每步+1
        let binding = ShapeBinding::new().bind("total_seq", total_seq);
        let resolved = total_seq_dim.resolve(&binding).unwrap();
        assert_eq!(resolved, total_seq,
            "step {step}: expected total_seq={total_seq}, got {resolved}");
    }
}
