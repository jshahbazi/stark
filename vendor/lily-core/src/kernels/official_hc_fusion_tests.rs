// SPDX-License-Identifier: Apache-2.0
// Synthetic controls for exact-boundary official HC read fusion.
// Root runs on Metal. Failure diagnostics contain aggregates only.
use super::*;
use crate::kernels::skinny::gemm_skinny_q8_nt;
use crate::moe_ffn::project_mat;

fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn values(state: &mut u32, count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|_| ((word(state) % 4096) as f32 / 2048.0 - 1.0) * scale).collect()
}

fn q8(ctx: &MetalContext, state: &mut u32, n: usize, k: usize) -> Result<QuantWeights> {
    let allocated = n + 2;
    let codes: Vec<u32> = (0..allocated * k / 4).map(|_| word(state)).collect();
    let scale: Vec<f32> = (0..allocated * k / 64)
        .map(|_| 0.0005 + (word(state) % 4096) as f32 / 4096.0 * 0.0095)
        .collect();
    let bias = values(state, allocated * k / 64, 1.0);
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[allocated, k / 4],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scale, &[allocated, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &bias, &[allocated, k / 64])?,
        group_size: 64,
        bits: 8,
    }
    .view_rows(2, n)
}

struct Guarded {
    allocation: Tensor,
    value: Tensor,
}

impl Guarded {
    fn new(ctx: &MetalContext, n: usize) -> Result<Self> {
        let allocation = Tensor::zeros(ctx, &[n + 8], DType::BF16)?;
        allocation.write_bytes(&vec![0xa5; allocation.byte_len()])?;
        let value = allocation.view(4, &[1, n])?;
        Ok(Self { allocation, value })
    }

    fn check(&self) -> Result<()> {
        let raw = self.allocation.raw_bytes();
        ensure!(
            raw[..8].iter().chain(raw[raw.len() - 8..].iter()).all(|b| *b == 0xa5),
            "official HC fusion wrote outside output view"
        );
        Ok(())
    }
}

fn exact(a: &Tensor, b: &Tensor, category: &str) -> Result<()> {
    let a = a.raw_bytes();
    let b = b.raw_bytes();
    let mismatches =
        a.chunks_exact(2).zip(b.chunks_exact(2)).filter(|(a, b)| a != b).count();
    ensure!(
        a.len() == b.len() && mismatches == 0,
        "{category}: actual_bytes={}, reference_bytes={}, bf16_mismatches={mismatches}",
        a.len(),
        b.len(),
    );
    Ok(())
}

#[test]
fn official_fused_down_activation_matches_materialized_q8_with_rounding_witness()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let mut state = 0x4843_4644;
    let weight = q8(&ctx, &mut state, 320, 10240)?;
    let input =
        Tensor::from_f32_as_bf16(&ctx, &values(&mut state, 10240, 3.0), &[1, 10240])?;
    let down = Guarded::new(&ctx, 320)?;
    let act = Guarded::new(&ctx, 320)?;
    let ref_down = Tensor::zeros(&ctx, &[1, 320], DType::BF16)?;
    let ref_act = Tensor::zeros(&ctx, &[1, 320], DType::BF16)?;
    let direct_skinny = Tensor::zeros(&ctx, &[1, 320], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    let unused = Tensor::zeros(&ctx, &[1], DType::BF16)?;
    project_mat(&ctx, &pass, &input, &weight, &ref_down, &unused)?;
    gemm_skinny_q8_nt(&ctx, &pass, &input, &weight, &direct_skinny)?;
    pass.level_barrier(&[&ref_down])?;
    silu_scaled_bf16_with_rounding(
        &ctx, &pass, &ref_down, &ref_act, 0.25, true, false,
    )?;
    official_hc_down_act_q8(&ctx, &pass, &weight, &input, &down.value, &act.value)?;
    pass.commit()?.wait()?;
    exact(&direct_skinny, &ref_down, "project_mat down route changed")?;
    exact(&down.value, &ref_down, "official fused down")?;
    exact(&act.value, &ref_act, "official fused activation")?;
    let projected = ref_down.to_f32()?;
    let one_round: Vec<f32> = projected
        .iter()
        .map(|x| {
            let scaled = *x * 0.25;
            bf(scaled / (1.0 + (-scaled).exp()))
        })
        .collect();
    ensure!(
        ref_act
            .to_f32()?
            .iter()
            .zip(one_round)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "activation fixture did not distinguish materialized sigmoid rounding"
    );
    down.check()?;
    act.check()
}

#[test]
fn official_fused_up_mix_matches_materialized_q8_with_bf16_product_witness()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let mut state = 0x4843_4655;
    let weight = q8(&ctx, &mut state, 10240, 320)?;
    let activation =
        Tensor::from_f32_as_bf16(&ctx, &values(&mut state, 320, 0.75), &[1, 320])?;
    let hn =
        Tensor::from_f32_as_bf16(&ctx, &values(&mut state, 10240, 2.0), &[1, 10240])?;
    let logits = Tensor::zeros(&ctx, &[1, 10240], DType::BF16)?;
    let direct_skinny = Tensor::zeros(&ctx, &[1, 10240], DType::BF16)?;
    let reference = Tensor::zeros(&ctx, &[1, 2560], DType::BF16)?;
    let fused = Guarded::new(&ctx, 2560)?;
    let pass = ctx.begin_concurrent()?;
    let unused = Tensor::zeros(&ctx, &[1], DType::BF16)?;
    project_mat(&ctx, &pass, &activation, &weight, &logits, &unused)?;
    gemm_skinny_q8_nt(&ctx, &pass, &activation, &weight, &direct_skinny)?;
    pass.level_barrier(&[&logits])?;
    hc_mix_bf16_with_rounding(
        &ctx, &pass, &logits, &hn, &reference, 2560, 4, true, false, false,
    )?;
    official_hc_up_mix_q8(&ctx, &pass, &weight, &activation, &hn, &fused.value)?;
    pass.commit()?.wait()?;
    exact(&direct_skinny, &logits, "project_mat up route changed")?;
    exact(&fused.value, &reference, "official fused up/mix")?;
    let l = logits.to_f32()?;
    let x = hn.to_f32()?;
    let wrong: Vec<f32> = (0..2560)
        .map(|i| {
            bf((0..4)
                .map(|g| {
                    let j = g * 2560 + i;
                    (1.0 / (1.0 + (-l[j]).exp())) * x[j]
                })
                .sum::<f32>()
                * 0.25)
        })
        .collect();
    ensure!(
        reference.to_f32()?.iter().zip(wrong).any(|(a, b)| a.to_bits() != b.to_bits()),
        "mix fixture did not distinguish BF16 gate/product/mean boundaries"
    );
    fused.check()
}

#[test]
fn official_fused_read_keeps_materialized_two_round_grouped_norm_and_geometry()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let (h, g, r, k) = (2560, 4, 320, 10240);
    let mut state = 0x4843_464e;
    let down_w = q8(&ctx, &mut state, r, k)?;
    let up_w = q8(&ctx, &mut state, k, r)?;
    let hyper_values: Vec<f32> =
        (0..k).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let gains: Vec<f32> = (0..k).map(|i| bf(0.5 + (i % 192) as f32 / 128.0)).collect();
    let hyper = Tensor::from_f32_as_bf16(&ctx, &hyper_values, &[1, k])?;
    let norm = Tensor::from_f32_as_bf16(&ctx, &gains, &[k])?;
    let hn = Tensor::zeros(&ctx, &[1, k], DType::BF16)?;
    let down = Tensor::zeros(&ctx, &[1, r], DType::BF16)?;
    let act = Tensor::zeros(&ctx, &[1, r], DType::BF16)?;
    let up = Tensor::zeros(&ctx, &[1, k], DType::BF16)?;
    let reference = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let fused_down = Guarded::new(&ctx, r)?;
    let fused_act = Guarded::new(&ctx, r)?;
    let fused_mix = Guarded::new(&ctx, h)?;
    ensure!(
        official_narrow_sigmoid_supported(&norm, &down_w, &up_w, None, h, g, 1),
        "official final no-inject mixer was not admitted"
    );
    ensure!(
        !official_narrow_sigmoid_supported(&norm, &down_w, &up_w, None, h, g, 17),
        "non-narrow row count was admitted"
    );
    let wrong_input = Tensor::zeros(&ctx, &[2, k], DType::BF16)?;
    let wrong_act = Tensor::zeros(&ctx, &[2, r], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    ensure!(
        official_hc_down_act_q8(&ctx, &pass, &down_w, &wrong_input, &down, &act)
            .is_err(),
        "official fused down accepted multiple rows"
    );
    ensure!(
        official_hc_up_mix_q8(&ctx, &pass, &up_w, &wrong_act, &hn, &reference).is_err(),
        "official fused up accepted multiple rows"
    );
    rmsnorm_grouped_bf16_with_rounding(
        &ctx, &pass, &hyper, &norm, &hn, h, g, 0.0, 0.0, true,
    )?;
    pass.level_barrier(&[&hn])?;
    let unused = Tensor::zeros(&ctx, &[1], DType::BF16)?;
    project_mat(&ctx, &pass, &hn, &down_w, &down, &unused)?;
    pass.level_barrier(&[&down])?;
    silu_scaled_bf16_with_rounding(&ctx, &pass, &down, &act, 0.25, true, false)?;
    pass.level_barrier(&[&act])?;
    project_mat(&ctx, &pass, &act, &up_w, &up, &unused)?;
    pass.level_barrier(&[&up])?;
    hc_mix_bf16_with_rounding(
        &ctx, &pass, &up, &hn, &reference, h, g, true, false, false,
    )?;
    official_hc_down_act_q8(
        &ctx,
        &pass,
        &down_w,
        &hn,
        &fused_down.value,
        &fused_act.value,
    )?;
    pass.level_barrier(&[&fused_act.value])?;
    official_hc_up_mix_q8(&ctx, &pass, &up_w, &fused_act.value, &hn, &fused_mix.value)?;
    pass.commit()?.wait()?;
    exact(&fused_down.value, &down, "complete official fused down")?;
    exact(&fused_act.value, &act, "complete official fused activation")?;
    exact(&fused_mix.value, &reference, "complete official fused mix")?;
    let inv = 1.0 / 2.5f32.sqrt();
    let expected: Vec<f32> =
        hyper_values.iter().zip(&gains).map(|(x, w)| bf(bf(*x * inv) * w)).collect();
    let wrong: Vec<f32> =
        hyper_values.iter().zip(&gains).map(|(x, w)| bf(*x * inv * w)).collect();
    let actual = hn.to_f32()?;
    ensure!(
        actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "materialized official grouped normalization changed"
    );
    ensure!(
        expected.iter().zip(wrong).any(|(a, b)| a.to_bits() != b.to_bits()),
        "norm witness did not distinguish two-round normalization"
    );
    fused_down.check()?;
    fused_act.check()?;
    fused_mix.check()
}

#[test]
fn official_fused_down_tail_preserves_projection_rounding_and_signed_zero() -> Result<()>
{
    let ctx = MetalContext::new()?;
    let mut sums = vec![0.0f32; 32];
    sums[..16].copy_from_slice(&[
        0.0,
        -0.0,
        0.25,
        -0.25,
        0.375,
        -0.375,
        1.00390625,
        -1.00390625,
        1.00390637,
        -1.00390637,
        0.501953125,
        -0.501953125,
        4.015625,
        -4.015625,
        27.375,
        -27.375,
    ]);
    let input = Tensor::from_f32(&ctx, &sums, &[32])?;
    let materialized = Tensor::from_f32_as_bf16(&ctx, &sums, &[1, 32])?;
    let reference = Tensor::zeros(&ctx, &[1, 32], DType::BF16)?;
    let down = Tensor::zeros(&ctx, &[1, 32], DType::BF16)?;
    let act = Tensor::zeros(&ctx, &[1, 32], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    silu_scaled_bf16_with_rounding(
        &ctx,
        &pass,
        &materialized,
        &reference,
        0.25,
        true,
        false,
    )?;
    let pipeline = ctx.pipeline(
        "official_hc_down_tail_rounding_control",
        OFFICIAL_FUSION_SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[input.binding(), down.binding(), act.binding()],
        &[],
        Grid::Threads { grid: (32, 1, 1), threadgroup: (32, 1, 1) },
    )?;
    pass.commit()?.wait()?;
    exact(&down, &materialized, "adversarial projection BF16 boundary")?;
    exact(&act, &reference, "adversarial activation boundary")?;
    Ok(())
}
