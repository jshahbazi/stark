// SPDX-License-Identifier: Apache-2.0
// Stock arrays remain in RAM; print only hashes/counts/aggregate differences.
use super::*;
use crate::sha256::{hex, sha256};

const ORACLE_CONTROLLER: &str =
    "b50d842b3faf84b71fc2dfceaaa0f571a21f2279af3f6efb107d51b394d97fe8";
const SEED: usize = 0x384864;
const ACTUAL_STOCK_REPORT: &str =
    "d538e7f1431180b0515cc14abf58c1a79066f8998f900787a82d6fa6f6abf94a";
const ACTUAL_STOCK_OUTPUTS: [&str; 2] = [
    "ffe2bff0fe897d2b7f4ed08d4b5b8004c8c65b5d61a1f3d3305ba2d42028d459",
    "c4026ed6d8fa41c2151df73d60b377dd125e820d69d2da8af53d1637127796fc",
];

fn digest(raw: &[u8]) -> String {
    hex(&sha256(raw))
}
fn hash(t: &Tensor) -> String {
    digest(t.raw_bytes())
}
fn bf(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

#[test]
fn official_q8_stock_signed_offset_guard_rejects_without_allocating() {
    assert!(stock_signed_offsets_fit(16480, 2560));
    assert!(stock_signed_offsets_fit(2560, 6144));
    assert!(!stock_signed_offsets_fit(1 << 30, 2560));
    assert!(!stock_signed_offsets_fit(8, i32::MAX as usize / 4 + 1));
    assert!(!stock_signed_offsets_fit(usize::MAX, 2560));
    assert!(!stock_signed_offsets_fit(8, usize::MAX));
    assert!(!stock_signed_offsets_fit(0, 2560));
}

fn receipt() -> Result<serde_json::Value> {
    let path = std::env::var_os("STARK_QWEN38_Q8_PROJECTION_ORACLE")
        .ok_or_else(|| anyhow::anyhow!("set private stock Q8 projection receipt"))?;
    let expected = std::env::var("STARK_QWEN38_Q8_PROJECTION_ORACLE_SHA256")
        .map_err(|_| anyhow::anyhow!("set private stock Q8 receipt SHA256"))?;
    ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x)),
        "invalid private stock Q8 receipt SHA256"
    );
    let path = std::path::Path::new(&path);
    let meta = std::fs::metadata(path)
        .map_err(|_| anyhow::anyhow!("private Q8 receipt metadata unavailable"))?;
    ensure!(
        meta.is_file() && meta.len() <= 1024 * 1024,
        "invalid private Q8 receipt file"
    );
    let raw = std::fs::read(path)
        .map_err(|_| anyhow::anyhow!("private Q8 receipt unavailable"))?;
    ensure!(
        expected == ACTUAL_STOCK_REPORT && digest(&raw) == expected,
        "private Q8 receipt identity mismatch"
    );
    let v: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|_| anyhow::anyhow!("invalid private Q8 receipt JSON"))?;
    ensure!(
        v["schema"] == "stark-private-stock-q8-projection-oracle-v1"
            && v["status"] == "complete"
            && v["controller_sha256"] == ORACLE_CONTROLLER
            && v["device"] == "gpu"
            && v["model_loaded"] == false
            && v["raw_arrays_persisted"] == false
            && v["quantization_performed"] == false
            && v["checkpoint_read_or_modified"] == false
            && v["bits"] == 8
            && v["group_size"] == 64
            && v["mode"] == "affine"
            && v["transpose_weights"] == true
            && v["fixture_seed"] == SEED,
        "unqualified private stock Q8 receipt"
    );
    ensure!(
        v["stock_asset_sha256"]["lib/libmlx.dylib"]
            == "fc504cec0f86f4a20b111d60dd1112df178bfd7d9b7978d4d735fa1a9c9bf8d4"
            && v["stock_asset_sha256"]["lib/libmlxc.dylib"]
                == "83ebedc48c226376af608b939e17256550d4baa1ad0cc276195c676fdc55c501"
            && v["stock_asset_sha256"]["lib/mlx.metallib"]
                == "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1"
            && v["projection_source_sha256"]["mlx/backend/metal/kernels/quantized.h"]
                == "8e2ca65d8d8dafe67b97120d1127d54ce57032f621bdba161e54a0bc4a948b8c"
            && v["compiler_reference_receipt_sha256"]
                == "405bfdbcd720abd56a19f04dd8dfdce3c371f3cf61bf471dc25b12972dd8020d"
            && v["joined_input_row_order"] == serde_json::json!(["qkv", "z", "a", "b"])
            && v["joined_input_row_widths"] == serde_json::json!([10240, 6144, 48, 48]),
        "stock Q8 source/assets/join identity mismatch"
    );
    let rows = v["fixtures"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Q8 fixtures missing"))?;
    ensure!(rows.len() == 2, "Q8 fixture count");
    for (row, (name, k, n)) in rows.iter().zip([
        ("gdn_input_joined_qkv_z_a_b", 2560, 16480),
        ("gdn_out_projection", 6144, 2560),
    ]) {
        ensure!(
            row["fixture"] == name
                && row["M"] == 1
                && row["K"] == k
                && row["N"] == n
                && row["actual_output_dtype_is_bfloat16"] == true
                && row["repeat_mismatch_count"] == 0
                && row["output_finite_count"] == n
                && row["widened_f32_bf16_value_count"] == n,
            "unqualified stock Q8 fixture"
        );
    }
    Ok(v)
}

fn fixture(
    ctx: &MetalContext,
    k: usize,
    n: usize,
    gs: usize,
) -> Result<(QuantWeights, Tensor)> {
    let x: Vec<f32> = (0..k)
        .map(|i| bf((((i * 17 + SEED) % 71) as i32 - 35) as f32 / 23.0))
        .collect();
    let mut codes = Vec::with_capacity(n * k / 4);
    for row in 0..n {
        for word in 0..k / 4 {
            let mut packed = 0u32;
            for lane in 0..4 {
                let col = word * 4 + lane;
                packed |= (((row * 7 + col * 3 + col / 64 + SEED) % 256) as u32)
                    << (8 * lane);
            }
            codes.push(packed);
        }
    }
    let scales: Vec<f32> = (0..n)
        .flat_map(|r| {
            (0..k / gs)
                .map(move |g| bf(((r * 11 + g * 5 + SEED) % 23 + 1) as f32 / 512.0))
        })
        .collect();
    let biases: Vec<f32> = (0..n)
        .flat_map(|r| {
            (0..k / gs).map(move |g| {
                bf((((r * 3 + g * 7 + SEED) % 19) as i32 - 9) as f32 / 256.0)
            })
        })
        .collect();
    let mut code_guarded = vec![17u32; 4];
    code_guarded.extend(codes);
    code_guarded.extend([17; 4]);
    let w = QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&code_guarded),
            &[code_guarded.len()],
            DType::U32,
        )?
        .view(4, &[n, k / 4])?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[n, k / gs])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[n, k / gs])?,
        group_size: gs,
        bits: 8,
    };
    let mut guarded = vec![17.0f32; 4];
    guarded.extend(x);
    guarded.extend([17.0; 4]);
    let x = Tensor::from_f32_as_bf16(ctx, &guarded, &[guarded.len()])?.view(4, &[k])?;
    Ok((w, x))
}

#[test]
#[ignore = "requires physical Metal and private source-bound stock Q8 receipt plus SHA256"]
fn official_gdn_q8_decode_matches_exact_stock_joined_and_output() -> Result<()> {
    // Refuse missing/unqualified receipts before creating a Metal context.
    let oracle = receipt()?;
    let ctx = MetalContext::new()?;
    let mut matched = 0;
    for (index, (k, n)) in [(2560, 16480), (6144, 2560)].into_iter().enumerate() {
        let (w, x) = fixture(&ctx, k, n, 64)?;
        let row = &oracle["fixtures"][index];
        ensure!(
            row["output"]["sha256"] == ACTUAL_STOCK_OUTPUTS[index],
            "actual stock Q8 golden binding mismatch"
        );
        for (field, input) in [
            ("activation", &x),
            ("codes", &w.codes),
            ("scales", &w.scales),
            ("biases", &w.biases),
        ] {
            ensure!(
                row[field]["sha256"] == hash(input),
                "stock Q8 input hash mismatch"
            );
        }
        let storage = Tensor::from_f32_as_bf16(&ctx, &vec![17.0; n + 8], &[n + 8])?;
        let out = storage.view(4, &[n])?;
        let repeat = Tensor::zeros(&ctx, &[n], DType::BF16)?;
        let legacy = Tensor::zeros(&ctx, &[n], DType::BF16)?;
        let false_policy = Tensor::zeros(&ctx, &[n], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        gemv_gdn_decode(&ctx, &pass, &w, &x, &out, true)?;
        gemv_gdn_decode(&ctx, &pass, &w, &x, &repeat, true)?;
        quant::gemv_quant(&ctx, &pass, &w, &x, &legacy)?;
        gemv_gdn_decode(&ctx, &pass, &w, &x, &false_policy, false)?;
        pass.commit()?.wait()?;
        ensure!(hash(&out) == hash(&repeat), "Q8 repeat output hash mismatch");
        ensure!(
            hash(&legacy) == hash(&false_policy),
            "Q8 false policy changed legacy output"
        );
        let guarded = storage.to_f32()?;
        ensure!(
            guarded[..4].iter().chain(&guarded[n + 4..]).all(|x| *x == 17.0),
            "Q8 output canaries changed"
        );
        let actual_hash = hash(&out);
        let stock_match = row["output"]["sha256"] == actual_hash;
        matched += usize::from(stock_match);
        let a = out.to_f32()?;
        let b = legacy.to_f32()?;
        let changed =
            a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let max_abs =
            a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!(
            "official_gdn_q8_stock k={k} n={n} exact_stock_hash_match={stock_match} repeat_exact=true false_policy_exact=true legacy_changed_count={changed} legacy_max_abs={max_abs} actual_BF16_sha256={actual_hash}"
        );
    }
    ensure!(
        matched == 2,
        "strict stock Q8 output hashes: matched_cases={matched}/2; no tolerance substitution"
    );
    Ok(())
}

#[test]
#[ignore = "requires physical Metal"]
fn official_gdn_q8_nonadmitted_shapes_groups_and_output_types_preserve_legacy()
-> Result<()> {
    let ctx = MetalContext::new()?;
    for gs in [32, 64] {
        let (w, x) = fixture(&ctx, 256, 8, gs)?;
        for dtype in [DType::BF16, DType::F32] {
            let direct = Tensor::zeros(&ctx, &[8], dtype)?;
            let fallback = Tensor::zeros(&ctx, &[8], dtype)?;
            let disabled = Tensor::zeros(&ctx, &[8], dtype)?;
            let pass = ctx.begin_concurrent()?;
            quant::gemv_quant(&ctx, &pass, &w, &x, &direct)?;
            gemv_gdn_decode(&ctx, &pass, &w, &x, &fallback, true)?;
            gemv_gdn_decode(&ctx, &pass, &w, &x, &disabled, false)?;
            pass.commit()?.wait()?;
            ensure!(
                hash(&direct) == hash(&fallback) && hash(&direct) == hash(&disabled),
                "non-admitted Q8 route changed legacy output"
            );
        }
    }
    Ok(())
}
