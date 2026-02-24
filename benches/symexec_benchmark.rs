//! Benchmark symexec trace extraction for various scalar operators.
//!
//! REQ-SYMEXEC-003: symexec latency < 1ms per operator.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench symexec_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gllm_kernels::compiler::symexec::decoder::analyze_scalar_fn;
use gllm_kernels::compiler::trace::{ScalarFnSignature, ScalarParam};
use gllm_kernels::scalar_ops::activations::{scalar_gelu, scalar_relu, scalar_silu};

fn make_activation_sig(fn_ptr: *const u8) -> ScalarFnSignature {
    ScalarFnSignature {
        fn_ptr,
        params: vec![
            ScalarParam::InputPtr,
            ScalarParam::OutputPtr,
            ScalarParam::Dim(0),
        ],
    }
}

fn bench_symexec_silu(c: &mut Criterion) {
    let sig = make_activation_sig(scalar_silu as *const u8);
    c.bench_function("symexec_silu", |b| {
        b.iter(|| {
            let trace = analyze_scalar_fn(black_box(sig.fn_ptr), black_box(&sig));
            black_box(trace).unwrap();
        })
    });
}

fn bench_symexec_gelu(c: &mut Criterion) {
    let sig = make_activation_sig(scalar_gelu as *const u8);
    c.bench_function("symexec_gelu", |b| {
        b.iter(|| {
            let trace = analyze_scalar_fn(black_box(sig.fn_ptr), black_box(&sig));
            black_box(trace).unwrap();
        })
    });
}

fn bench_symexec_relu(c: &mut Criterion) {
    let sig = make_activation_sig(scalar_relu as *const u8);
    c.bench_function("symexec_relu", |b| {
        b.iter(|| {
            let trace = analyze_scalar_fn(black_box(sig.fn_ptr), black_box(&sig));
            black_box(trace).unwrap();
        })
    });
}

criterion_group!(
    name = symexec;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    targets = bench_symexec_silu, bench_symexec_gelu, bench_symexec_relu,
);
criterion_main!(symexec);
