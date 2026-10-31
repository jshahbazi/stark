// Original Stark synthetic controls. No model weights, prompts or outputs.
// These are native component candidates, not stock-MLX parity evidence.
use super::*;
use crate::sha256::{Sha256, hex};
use half::bf16;

fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn sig(x: f32) -> f32 {
    let x = bf(x);
    let ex = bf(x.abs().exp());
    let den = bf(1.0 + ex);
    let y = bf(1.0 / den);
    if x < 0.0 { y } else { bf(1.0 - y) }
}

fn exact(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(actual.len() == expected.len(), "component length mismatch");
    ensure!(
        actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "official MoE component bit mismatch"
    );
    Ok(())
}

#[test]
fn official_moe_actual_shape_guard_and_rounding_witnesses() {
    assert!(official_moe_shape_supported(512, 10, 2560, 640));
    for shape in [
        (256, 10, 2560, 640),
        (512, 8, 2560, 640),
        (512, 10, 4096, 640),
        (512, 10, 2560, 1024),
    ] {
        assert!(!official_moe_shape_supported(shape.0, shape.1, shape.2, shape.3));
    }
    // Scores multiply before each expert's BF16 output in the served A4
    // down reduction. The halfway projections deliberately distinguish it.
    let down = [1.00390625f32, 1.01171875];
    let scores = [0.25f32, 0.75];
    let served = bf(down[0] * scores[0] + down[1] * scores[1]);
    let legacy = bf(bf(down[0]) * scores[0] + bf(down[1]) * scores[1]);
    assert_ne!(served.to_bits(), legacy.to_bits());
    // Full-E BF16 softmax probabilities cannot be canceled algebraically
    // into a selected-only F32 softmax after the materialized round points.
    let mut logits = vec![0.0f32; 512];
    for (i, x) in logits.iter_mut().take(10).enumerate() {
        *x = bf(1.0 - i as f32 / 16.0);
    }
    let max = logits[0];
    let denominator: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    let picked: Vec<f32> =
        logits[..10].iter().map(|x| bf((x - max).exp() / denominator)).collect();
    let total = picked.iter().fold(0.0, |sum, x| bf(sum + x));
    let served: Vec<f32> = picked.iter().map(|p| bf(p / bf(total))).collect();
    let selected_den: f32 = logits[..10].iter().map(|x| (x - max).exp()).sum();
    let legacy: Vec<f32> =
        logits[..10].iter().map(|x| (x - max).exp() / selected_den).collect();
    assert!(served.iter().zip(legacy).any(|(a, b)| a.to_bits() != b.to_bits()));
    // These distinct F32 keys collapse to the same BF16 key: routing must
    // honor the smaller expert ID after the actual BF16 projection boundary.
    assert_eq!(bf(1.001).to_bits(), bf(1.002).to_bits());
    assert!(1.002f32 > 1.001f32);
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device; stock oracle still required"]
fn official_moe_sigmoid_table_all_finite_bf16() -> Result<()> {
    let ctx = MetalContext::new()?;
    let table = official_moe_sigmoid_table(&ctx)?;
    let actual = table.to_f32()?;
    let mut hash = Sha256::new();
    let mut checked = 0;
    for bits in 0..=u16::MAX {
        if bf16::from_bits(bits).to_f32().is_finite() {
            hash.update(&bf16::from_f32(actual[bits as usize]).to_bits().to_le_bytes());
            checked += 1;
        }
    }
    // Actual stock GPU table, not the host expf reconstruction. The latter
    // differs in three subnormal cases and is not an exhaustive authority.
    let actual_hash = hex(&hash.finish());
    ensure!(checked == 65280, "finite BF16 table domain mismatch");
    ensure!(
        actual_hash
            == "c487ccb3208e0ad603a280dd5b17a28910f9487929a0ba9bdfb1f782b3722b32",
        "native GPU LUT differs from the canonical stock GPU finite table"
    );
    eprintln!("official_moe_sigmoid_finite count={checked} sha256={actual_hash}");
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_moe_router_bf16_ties_and_normalization() -> Result<()> {
    let ctx = MetalContext::new()?;
    for rows in [1, 2, 33] {
        // All logits equal gives exactly representable 1/512 probabilities
        // and denominator 10/512; expected score is the BF16 1/10.
        let logits =
            Tensor::from_f32_as_bf16(&ctx, &vec![0.0; rows * 512], &[rows, 512])?;
        let ids = Tensor::zeros(&ctx, &[rows, 10], DType::U32)?;
        let scores = Tensor::zeros(&ctx, &[rows, 10], DType::F32)?;
        let pass = ctx.begin()?;
        official_moe_router(&ctx, &pass, &logits, &ids, &scores, true, None)?;
        pass.commit_wait()?;
        let actual_ids: &[u32] = bytemuck::cast_slice(ids.raw_bytes());
        ensure!(
            actual_ids.chunks_exact(10).all(|r| r == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "official router BF16 tie policy mismatch"
        );
        exact(&scores.to_f32()?, &vec![bf(0.1); rows * 10])?;
    }
    // The public router accepts BF16 only. F32 keys cannot bypass the
    // projection materialization and silently choose another tied expert.
    let raw = Tensor::from_f32(&ctx, &vec![0.0; 512], &[512])?;
    let ids = Tensor::zeros(&ctx, &[10], DType::U32)?;
    let scores = Tensor::zeros(&ctx, &[10], DType::F32)?;
    let pass = ctx.begin()?;
    ensure!(
        official_moe_router(&ctx, &pass, &raw, &ids, &scores, true, None).is_err(),
        "official router admitted F32 keys"
    );
    pass.commit_wait()?;
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_moe_router_folded_gate_matches_served_partition() -> Result<()> {
    let ctx = MetalContext::new()?;
    let logits = Tensor::from_f32_as_bf16(&ctx, &vec![0.0; 512], &[512])?;
    let ids = Tensor::zeros(&ctx, &[10], DType::U32)?;
    let scores = Tensor::zeros(&ctx, &[10], DType::F32)?;
    let xv: Vec<f32> = (0..2560).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
    let wv: Vec<f32> = (0..2560).map(|i| ((i % 13) as f32 - 6.0) / 32.0).collect();
    // Dyadic operands keep the float reduction exact while still exercising
    // every one of the 7 spare SIMDgroups and the predicated final stride.
    let expected = bf(xv.iter().zip(&wv).map(|(a, b)| a * b).sum());
    let x = Tensor::from_f32_as_bf16(&ctx, &xv, &[2560])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &wv, &[1, 2560])?;
    let gate = Tensor::zeros(&ctx, &[1], DType::BF16)?;
    let pass = ctx.begin()?;
    official_moe_router(
        &ctx,
        &pass,
        &logits,
        &ids,
        &scores,
        true,
        Some((&x, &w, &gate)),
    )?;
    pass.commit_wait()?;
    exact(&gate.to_f32()?, &[expected])
}

fn synthetic_bank(
    ctx: &MetalContext,
    experts: usize,
    n: usize,
    k: usize,
    bias: impl Fn(usize, usize, usize) -> f32,
) -> Result<QuantWeights> {
    let rows = experts * n;
    let groups = k / 64;
    let b: Vec<f32> = (0..rows * groups)
        .map(|at| {
            let row = at / groups;
            bias(row / n, row % n, at % groups)
        })
        .collect();
    Ok(QuantWeights {
        codes: Tensor::zeros(ctx, &[rows, k / 8], DType::U32)?,
        scales: Tensor::from_f32_as_bf16(
            ctx,
            &vec![1.0 / 1024.0; rows * groups],
            &[rows, groups],
        )?,
        biases: Tensor::from_f32_as_bf16(ctx, &b, &[rows, groups])?,
        bits: 4,
        group_size: 64,
    })
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_moe_gate_up_rounds_silu_and_two_products() -> Result<()> {
    let ctx = MetalContext::new()?;
    let table = official_moe_sigmoid_table(&ctx)?;
    let pattern = [-6.84375f32, -4.0, 0.375, 2.5];
    // Same actual H2560/I640 and x16 per-lane dot geometry; a smaller
    // synthetic expert bank suffices to isolate arithmetic without a model.
    let gate = synthetic_bank(&ctx, 4, 640, 2560, |e, r, g| {
        if g == 0 { pattern[(e + r) % 4] / 64.0 } else { 0.0 }
    })?;
    let up =
        synthetic_bank(
            &ctx,
            4,
            640,
            2560,
            |_, _, g| if g == 0 { 1.25 / 64.0 } else { 0.0 },
        )?;
    let mut xv = vec![0.0; 2560];
    xv[..64].fill(1.0);
    let x = Tensor::from_f32_as_bf16(&ctx, &xv, &[2560])?;
    let iv: Vec<u32> = (0..10).map(|i| (i % 4) as u32).collect();
    let ids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&iv), &[10], DType::U32)?;
    let out = Tensor::zeros(&ctx, &[10, 640], DType::BF16)?;
    let pass = ctx.begin()?;
    let pipeline =
        ctx.pipeline("official_moe_gate_up_a4", OFFICIAL_MOE_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            ids.binding(),
            table.binding(),
            out.binding(),
        ],
        &[&u32_bytes(2560), &u32_bytes(640), &u32_bytes(64)],
        Grid::Threadgroups { groups: (320, 10, 1), threadgroup: (64, 1, 1) },
    )?;
    pass.commit_wait()?;
    let expected: Vec<f32> = (0..10 * 640)
        .map(|i| {
            let g = pattern[((i / 640) % 4 + i % 640) % 4];
            bf(bf(g * sig(g)) * 1.25)
        })
        .collect();
    let old: Vec<f32> = (0..10 * 640)
        .map(|i| {
            let g = pattern[((i / 640) % 4 + i % 640) % 4];
            bf((g / (1.0 + (-g).exp())) * 1.25)
        })
        .collect();
    ensure!(
        old.iter().zip(&expected).any(|(a, b)| a.to_bits() != b.to_bits()),
        "gate/up fixture does not distinguish activation boundaries"
    );
    exact(&out.to_f32()?, &expected)
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_moe_down_weights_before_expert_round_and_shared_bf16_tail() -> Result<()> {
    let ctx = MetalContext::new()?;
    let w = synthetic_bank(&ctx, 2, 2560, 640, |e, _, g| match g {
        0 => 1.0 / 64.0,
        1 => {
            if e == 0 {
                1.0 / 16384.0
            } else {
                3.0 / 16384.0
            }
        }
        _ => 0.0,
    })?;
    let mut xv = vec![0.0; 10 * 640];
    for e in 0..10 {
        xv[e * 640..e * 640 + 128].fill(1.0);
    }
    let x = Tensor::from_f32_as_bf16(&ctx, &xv, &[10, 640])?;
    let mut iv = vec![0u32; 10];
    iv[1] = 1;
    let ids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&iv), &[10], DType::U32)?;
    let mut sv = vec![0.0; 10];
    sv[0] = 0.25;
    sv[1] = 0.75;
    let scores = Tensor::from_f32(&ctx, &sv, &[10])?;
    let shared = Tensor::from_f32_as_bf16(&ctx, &vec![0.375; 2560], &[2560])?;
    let gl = Tensor::from_f32_as_bf16(&ctx, &[-6.84375], &[1])?;
    let out = Tensor::zeros(&ctx, &[2560], DType::BF16)?;
    let routed = bf(0.25 * 1.00390625 + 0.75 * 1.01171875);
    let old = bf(0.25 * bf(1.00390625) + 0.75 * bf(1.01171875));
    ensure!(
        routed.to_bits() != old.to_bits(),
        "down fixture lost its boundary witness"
    );
    for has_shared in [false, true] {
        let pass = ctx.begin()?;
        let pipeline = ctx.pipeline(
            "official_moe_down_a4",
            OFFICIAL_MOE_SOURCE,
            MslVersion::V3_1,
        )?;
        pass.dispatch_at(
            &pipeline,
            &[
                w.codes.binding(),
                w.scales.binding(),
                w.biases.binding(),
                x.binding(),
                ids.binding(),
                scores.binding(),
                shared.binding(),
                gl.binding(),
                out.binding(),
            ],
            &[
                &u32_bytes(640),
                &u32_bytes(2560),
                &u32_bytes(64),
                &u32_bytes(10),
                &u32_bytes(has_shared as usize),
            ],
            Grid::Threadgroups { groups: (1280, 1, 1), threadgroup: (64, 1, 1) },
        )?;
        pass.commit_wait()?;
        let expected =
            if has_shared { bf(routed + bf(sig(-6.84375) * 0.375)) } else { routed };
        exact(&out.to_f32()?, &vec![expected; 2560])?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_moe_prefill_eight_partials_and_short_tails() -> Result<()> {
    let ctx = MetalContext::new()?;
    for m in [1usize, 2, 8, 17, 33] {
        let h = 2560;
        let ev: Vec<f32> =
            (0..m * 10 * h).map(|i| bf(((i % 37) as f32 - 18.0) / 16.0)).collect();
        let sv: Vec<f32> =
            (0..m * 10).map(|i| bf((i % 10 + 1) as f32 / 64.0)).collect();
        let ids: Vec<u32> = (0..m * 10).map(|i| (m * 10 - 1 - i) as u32).collect();
        let sharedv: Vec<f32> =
            (0..m * h).map(|i| bf(((i % 11) as f32 - 5.0) / 32.0)).collect();
        let gv: Vec<f32> = (0..m).map(|i| [-6.84375, -1.5, 2.5][i % 3]).collect();
        let ed = Tensor::from_f32_as_bf16(&ctx, &ev, &[m * 10, h])?;
        let slots =
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[m, 10], DType::U32)?;
        let scores = Tensor::from_f32(&ctx, &sv, &[m, 10])?;
        let shared = Tensor::from_f32_as_bf16(&ctx, &sharedv, &[m, h])?;
        let gate = Tensor::from_f32_as_bf16(&ctx, &gv, &[m])?;
        let backing = Tensor::zeros(&ctx, &[m * h + 16], DType::BF16)?;
        backing.write_bytes(&vec![0xa5; backing.byte_len()])?;
        let out = backing.view(8, &[m, h])?;
        let pass = ctx.begin()?;
        official_moe_prefill_combine(
            &ctx, &pass, &ed, &slots, &scores, &shared, &gate, &out,
        )?;
        pass.commit_wait()?;
        let expected: Vec<f32> = (0..m * h)
            .map(|at| {
                let row = at / h;
                let col = at % h;
                let mut parts = [0.0f32; 8];
                for e in 0..10 {
                    let slot = ids[row * 10 + e] as usize;
                    let product = bf(ev[slot * h + col] * sv[row * 10 + e]);
                    parts[e % 8] = bf(product + parts[e % 8]);
                }
                let total = parts[1..].iter().fold(parts[0], |sum, p| bf(p + sum));
                bf(total + bf(sig(gv[row]) * sharedv[at]))
            })
            .collect();
        exact(&out.to_f32()?, &expected)?;
        let raw = backing.raw_bytes();
        ensure!(
            raw[..16].iter().chain(&raw[raw.len() - 16..]).all(|b| *b == 0xa5),
            "official prefill combine wrote beyond output view"
        );
    }
    Ok(())
}
