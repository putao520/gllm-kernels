#[test]
fn check_isa_level() {
    use crate::cpu_kernels::{get_isa_level, IsaLevel};
    let isa = get_isa_level();
    println!("Detected ISA Level: {:?}", isa);
    match isa {
        IsaLevel::Avx2 => assert!(true),
        _ => println!("WARNING: AVX2 not detected. Benchmark results will be scalar."),
    }
}
