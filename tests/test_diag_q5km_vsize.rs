#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn q5km_v_size() {
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q5_k_m").build().expect("c");
    let o = c.diagnostic_weight_offsets().expect("o");
    // L0.v_proj (Q6K) → L0.q_norm 之间的大小
    let v = o.iter().find(|(n,_,_)| n=="L0.v_proj").map(|(_,o,_)| *o);
    let qn = o.iter().find(|(n,_,_)| n=="L0.q_norm").map(|(_,o,_)| *o);
    eprintln!("L0.v_proj={:?} L0.q_norm={:?} diff={:?}", v, qn, qn.zip(v).map(|(a,b)|a-b));
    // L0.q_proj (Q5K) → L0.k_proj
    let qp = o.iter().find(|(n,_,_)| n=="L0.q_proj").map(|(_,o,_)| *o);
    let kp = o.iter().find(|(n,_,_)| n=="L0.k_proj").map(|(_,o,_)| *o);
    eprintln!("L0.q_proj={:?} L0.k_proj={:?} diff={:?}", qp, kp, kp.zip(qp).map(|(a,b)|a-b));
    // L0.gate_proj (Q5K) → L0.up_proj → L0.down_proj (Q6K)
    let gp = o.iter().find(|(n,_,_)| n=="L0.gate_proj").map(|(_,o,_)| *o);
    let up = o.iter().find(|(n,_,_)| n=="L0.up_proj").map(|(_,o,_)| *o);
    let dp = o.iter().find(|(n,_,_)| n=="L0.down_proj").map(|(_,o,_)| *o);
    let pn = o.iter().find(|(n,_,_)| n=="L0.post_attn_norm").map(|(_,o,_)| *o);
    eprintln!("gate={:?} up={:?} diff={} down={:?} diff={} post_attn={:?}", gp, up, up.zip(gp).map(|(a,b)|a-b), dp, dp.zip(up).map(|(a,b)|a-b), pn);
}
