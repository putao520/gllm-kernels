use criterion::{criterion_group, criterion_main, Criterion};

fn attention_bench(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| 42u64));
}

criterion_group!(benches, attention_bench);
criterion_main!(benches);
