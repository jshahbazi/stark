// Original Stark controls against private, source-bound stock MLX receipts.
// No model data is opened. Synthetic arrays stay in RAM; compare hashes only.
use super::*;
use crate::sha256::{Sha256, hex};
use half::bf16;

fn digest(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finish())
}
fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn bf_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes()).collect()
}
fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn fixtures() -> Vec<(&'static str, Vec<f32>)> {
    let mut out = Vec::new();
    for (name, seed) in [("nonuniform_a", 0x4d4f4531u32), ("nonuniform_b", 0x6a3912d1)]
    {
        let mut state = seed;
        out.push((
            name,
            (0..512)
                .map(|_| ((next(&mut state) % 1024) as i32 - 512) as f32 / 128.0)
                .collect(),
        ));
    }
    let mut v = vec![-2.0; 512];
    for (i, x) in v.iter_mut().take(12).enumerate() {
        *x = 1.0 + (12 - i) as f32 / 8192.0;
    }
    out.push(("bf16_boundary_tie", v));
    let mut v = vec![0.0; 512];
    for (i, x) in v.iter_mut().take(9).enumerate() {
        *x = 2.0 - i as f32 / 32.0;
    }
    v[9] = 1.001;
    v[10] = 1.002;
    out.push(("cutoff_boundary_tie", v));
    for (name, base) in [("full_axis_base", -2.0), ("full_axis_shift", -1.0)] {
        let mut v: Vec<f32> = (0..512).map(|i| base + (i % 7) as f32 / 128.0).collect();
        for (i, x) in v.iter_mut().take(10).enumerate() {
            *x = 1.0 - i as f32 / 16.0;
        }
        out.push((name, v));
    }
    out
}

fn receipt() -> Result<serde_json::Value> {
    let path = std::env::var_os("STARK_QWEN38_MOE_ORACLE")
        .ok_or_else(|| anyhow::anyhow!("set private stock MoE receipt path"))?;
    let path = std::path::Path::new(&path);
    let meta = std::fs::metadata(path)?;
    ensure!(
        meta.is_file() && meta.len() <= 1024 * 1024,
        "invalid private MoE receipt file"
    );
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        value["schema"] == "stark-private-stock-moe-oracle-v1"
            && value["status"] == "complete"
            && value["device"] == "gpu"
            && value["model_loaded"] == false
            && value["raw_arrays_persisted"] == false,
        "unqualified stock MoE receipt"
    );
    ensure!(
        value["native_fixture_sha256"]
            == digest(include_bytes!("official_moe_oracle_tests.rs")),
        "stock MoE native fixture identity mismatch"
    );
    ensure!(
        value["native_shader_sha256"]
            == digest(include_bytes!("metal/official_moe.metal")),
        "stock MoE native shader identity mismatch"
    );
    ensure!(
        value["transformer_sha256"]
            == "b2879bdd3f15bef2607df9aeb205977468fccde0e5f3d9a2ba1010384fe53489"
            && value["stock_asset_sha256"]["lib/libmlx.dylib"]
                == "fc504cec0f86f4a20b111d60dd1112df178bfd7d9b7978d4d735fa1a9c9bf8d4"
            && value["stock_asset_sha256"]["lib/libmlxc.dylib"]
                == "83ebedc48c226376af608b939e17256550d4baa1ad0cc276195c676fdc55c501"
            && value["stock_asset_sha256"]["lib/mlx.metallib"]
                == "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1",
        "stock MoE reference identity mismatch"
    );
    Ok(value)
}

#[test]
#[ignore = "requires physical Metal and a source-bound stock MoE C-API receipt"]
fn official_moe_device_router_matches_stock_nonuniform_and_bf16_boundaries()
-> Result<()> {
    let oracle = receipt()?;
    ensure!(
        oracle["routing_alternative_witness_count"].as_u64().unwrap_or(0) > 0,
        "stock fixture does not distinguish full-axis from selected-only BF16 routing"
    );
    let ctx = MetalContext::new()?;
    let xv: Vec<f32> = (0..2560).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
    let wv: Vec<f32> = (0..2560).map(|i| ((i % 13) as f32 - 6.0) / 32.0).collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &xv, &[2560])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &wv, &[1, 2560])?;
    for (name, values) in fixtures() {
        let record = oracle["router"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing router receipt"))?
            .iter()
            .find(|r| r["name"] == name)
            .ok_or_else(|| anyhow::anyhow!("missing router fixture"))?;
        ensure!(
            record["input_bf16_sha256"] == digest(&bf_bytes(&values)),
            "router input identity mismatch"
        );
        let mut expected: Vec<u32> = (0..512).collect();
        expected.sort_by(|a, b| {
            bf(values[*b as usize]).total_cmp(&bf(values[*a as usize])).then(a.cmp(b))
        });
        expected.truncate(10);
        let logits = Tensor::from_f32_as_bf16(&ctx, &values, &[512])?;
        let ids = Tensor::zeros(&ctx, &[10], DType::U32)?;
        let scores = Tensor::zeros(&ctx, &[10], DType::F32)?;
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
        ensure!(
            ids.raw_bytes() == bytemuck::cast_slice::<u32, u8>(&expected),
            "device BF16 boundary/tie IDs differ"
        );
        ensure!(
            digest(ids.raw_bytes()) == record["custom_indices_sha256"],
            "native/stock router IDs differ"
        );
        let actual = scores.to_f32()?;
        ensure!(
            actual.iter().all(|s| s.to_bits() & 0xffff == 0),
            "official router stored a non-BF16 score"
        );
        ensure!(
            digest(&bf_bytes(&actual)) == record["custom_scores_sha256"],
            "native/stock router scores differ"
        );
        ensure!(
            digest(gate.raw_bytes()) == record["custom_gate_sha256"],
            "native/stock folded shared gate differs"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires physical Metal and a source-bound stock MoE C-API receipt"]
fn official_moe_device_prefill_matches_actual_stock_multiply_sum_axis() -> Result<()> {
    let oracle = receipt()?;
    let ctx = MetalContext::new()?;
    for m in [2usize, 8, 17, 33] {
        let h = 2560;
        let ev: Vec<f32> =
            (0..m * 10 * h).map(|i| bf(((i % 37) as f32 - 18.0) / 16.0)).collect();
        let sv: Vec<f32> =
            (0..m * 10).map(|i| bf((i % 10 + 1) as f32 / 64.0)).collect();
        let ids: Vec<u32> = (0..m * 10).map(|i| (m * 10 - 1 - i) as u32).collect();
        let mut input = bf_bytes(&ev);
        input.extend_from_slice(bytemuck::cast_slice::<u32, u8>(&ids));
        input.extend_from_slice(&bf_bytes(&sv));
        let record = oracle["prefill"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing prefill receipt"))?
            .iter()
            .find(|r| r["m"] == m)
            .ok_or_else(|| anyhow::anyhow!("missing prefill fixture"))?;
        ensure!(
            record["input_sha256"] == digest(&input),
            "prefill input identity mismatch"
        );
        ensure!(
            record["custom_vs_primitive_mismatch_count"] == 0,
            "stock custom/primitive reduction differs"
        );
        let ed = Tensor::from_f32_as_bf16(&ctx, &ev, &[m * 10, h])?;
        let slots =
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[m, 10], DType::U32)?;
        let scores = Tensor::from_f32(&ctx, &sv, &[m, 10])?;
        let shared = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let gate = Tensor::zeros(&ctx, &[m], DType::BF16)?;
        let out = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let pass = ctx.begin()?;
        official_moe_prefill_combine(
            &ctx, &pass, &ed, &slots, &scores, &shared, &gate, &out,
        )?;
        pass.commit_wait()?;
        ensure!(
            digest(out.raw_bytes()) == record["stock_primitive_output_sha256"],
            "native combine differs from actual stock multiply/sum-axis"
        );
    }
    Ok(())
}
