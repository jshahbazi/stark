// SPDX-License-Identifier: Apache-2.0
// Test-only dense TAB1 diagnostic. Only hashes, shapes and aggregate flags leave
// this module. A mismatch report is persisted before the numerical assertion.
use std::{
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};

use super::*;
use crate::sha256::{hex, sha256};

const HK: usize = 16;
const HV: usize = 48;
const D: usize = 128;
const C: usize = 10240;
const ROWS: [usize; 2] = [65, 129];
const CONTROLLER_SHA: &str =
    "53659326ca1aa9d1eea2171a50f13e507b7ca1e84883604cee110cf5d27cfaa9";
const SPEC_SHA: &str =
    "b42940e47610af462fc17ec2347529689b35547560deb88fa9f827ed39edbb34";
const GDN_METAL_SHA: &str =
    "78ca01774e89e1fea05190234171704cc6a2bca4b24415cee2aa45cdd5ac6844";
const TB12_SHA: &str =
    "2c099d5cb38b0ba3dd819762554f0fa9ce91e8709dd5b3815d9423d2b73e5313";
const SCOPE: &str =
    "S65/S129 TAB1 only; no short/TAB0/capture/vector-gate/speculation qualification";
const CASES: [u16; 33] = [
    0xc0db, 0x0000, 0x8000, 0x3ebf, 0xbebf, 0x3ec0, 0xbec0, 0x3ec1, 0xbec1, 0x3f9f,
    0xbf9f, 0x3fa0, 0xbfa0, 0x3fff, 0xbfff, 0x4000, 0xc000, 0x4001, 0xc001, 0x401f,
    0xc01f, 0x4020, 0xc020, 0x4021, 0xc021, 0x40ff, 0xc0ff, 0x4100, 0xc100, 0x4180,
    0xc180, 0x42a0, 0xc2a0,
];

fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}
fn bf_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
fn digest(bytes: &[u8]) -> String {
    hex(&sha256(bytes))
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("dense oracle string field"))
}
fn hash_is(bytes: &[u8], expected: &str) -> Result<()> {
    ensure!(digest(bytes) == expected, "dense oracle input/source hash guard");
    Ok(())
}
fn sha_text(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

struct Fixture {
    rows: usize,
    salt: usize,
    qkv: Vec<f32>,
    history: Vec<f32>,
    weights: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    state: Vec<f32>,
    z: Vec<f32>,
    gamma: Vec<f32>,
    probe_conv: Vec<f32>,
    probe_y: Vec<f32>,
}

fn fixture(rows: usize) -> Fixture {
    let salt = rows % 17;
    let mut qkv: Vec<_> = (0..rows * C)
        .map(|i| bf((((i * 13 + salt * 7) % 113) as i32 - 56) as f32 / 53.0f32))
        .collect();
    for row in 0..rows {
        for ch in 0..CASES.len() {
            qkv[row * C + ch] =
                half::bf16::from_bits(CASES[(ch + row) % CASES.len()]).to_f32();
        }
    }
    let mut weights: Vec<_> = (0..C)
        .flat_map(|ch| {
            (0..4).map(move |tap| {
                bf((((ch * 3 + tap * 11 + salt) % 31) as i32 - 15) as f32 / 37.0f32)
            })
        })
        .collect();
    for ch in 0..CASES.len() {
        for tap in 0..4 {
            weights[ch * 4 + tap] = if tap == 3 { 1.0 } else { 0.0 };
        }
    }
    Fixture {
        rows,
        salt,
        qkv,
        weights,
        history: (0..3 * C)
            .map(|i| bf((((i * 7 + 3 + salt) % 59) as i32 - 29) as f32 / 47.0f32))
            .collect(),
        a: (0..rows * HV)
            .map(|i| bf((((i * 5 + salt) % 43) as i32 - 21) as f32 / 7.0f32))
            .collect(),
        b: (0..rows * HV)
            .map(|i| {
                half::bf16::from_bits(CASES[(i * 7 + salt) % CASES.len()]).to_f32()
            })
            .collect(),
        a_log: (0..HV).map(|h| bf((h as i32 % 9 - 4) as f32 / 13.0f32)).collect(),
        dt: (0..HV)
            .map(|h| bf((((h * 3 + salt) % 11) as i32 - 5) as f32 / 17.0f32))
            .collect(),
        state: (0..HV * D * D)
            .map(|i| {
                let first = (((i * 17 + 3) % 257) as i32 - 128) as f32 / 997.0f32;
                let second = (i as i32 % 11 - 5) as f32 / 65537.0f32;
                first + second
            })
            .collect(),
        z: (0..rows * HV * D)
            .map(|i| {
                half::bf16::from_bits(CASES[(i * 13 + salt) % CASES.len()]).to_f32()
            })
            .collect(),
        gamma: (0..D)
            .map(|i| bf(1.0f32 + (i as i32 % 7 - 3) as f32 / 17.0f32))
            .collect(),
        probe_conv: (0..rows * C)
            .map(|i| bf((((i * 17 + salt) % 131) as i32 - 65) as f32 / 97.0f32))
            .collect(),
        probe_y: (0..rows * HV * D)
            .map(|i| bf((((i * 19 + 5 + salt) % 139) as i32 - 69) as f32 / 101.0f32))
            .collect(),
    }
}

impl Fixture {
    fn input_hashes(&self) -> Value {
        let mut result = serde_json::Map::new();
        for (name, values) in [
            ("qkv", &self.qkv),
            ("history", &self.history),
            ("weights", &self.weights),
            ("a", &self.a),
            ("b", &self.b),
            ("dt", &self.dt),
            ("z", &self.z),
            ("gamma", &self.gamma),
            ("probe_conv", &self.probe_conv),
            ("probe_y", &self.probe_y),
        ] {
            result.insert(
                format!("{name}_bf16_sha256"),
                Value::from(digest(&bf_bytes(values))),
            );
        }
        for (name, values) in
            [("a_log", &self.a_log), ("state", &self.state), ("gamma", &self.gamma)]
        {
            result.insert(
                format!("{name}_f32_sha256"),
                Value::from(digest(&f32_bytes(values))),
            );
        }
        Value::Object(result)
    }
}

fn descriptor_guard(value: &Value, shape: &[usize], dtype: &str) -> Result<()> {
    let expected_shape =
        shape.iter().map(|n| Value::from(*n as u64)).collect::<Vec<_>>();
    let width = if dtype == "bfloat16" { 2 } else { 4 };
    ensure!(
        value["shape"].as_array() == Some(&expected_shape),
        "dense oracle descriptor shape"
    );
    ensure!(value["dtype"].as_str() == Some(dtype), "dense oracle descriptor dtype");
    ensure!(
        value["bytes"].as_u64()
            == Some((shape.iter().product::<usize>() * width) as u64),
        "dense oracle descriptor bytes"
    );
    ensure!(sha_text(text(value, "sha256")?), "dense oracle descriptor hash syntax");
    Ok(())
}

fn case_guards(case: &Value, fixture: &Fixture) -> Result<()> {
    let m = fixture.rows;
    ensure!(
        text(case, "fixture")? == "dense_tab1_fullchain",
        "dense oracle fixture type"
    );
    ensure!(
        case["rows"].as_u64() == Some(m as u64)
            && case["salt"].as_u64() == Some(fixture.salt as u64),
        "dense oracle fixture row/salt"
    );
    ensure!(case["TAB"].as_u64() == Some(1), "dense oracle TAB route");
    ensure!(
        case["inputs"] == fixture.input_hashes(),
        "dense oracle all thirteen fixture input hashes"
    );
    ensure!(
        case["tap_original_output_mismatch_counts"] == json!([0, 0, 0, 0, 0, 0]),
        "dense oracle tap changed served prework"
    );
    ensure!(
        case["all_stage_repeats_exact"].as_bool() == Some(true),
        "dense oracle stock repeat"
    );
    ensure!(
        case["recurrence_comparison_requires_matching_prework_inputs"].as_bool()
            == Some(true),
        "dense oracle recurrence scope"
    );
    ensure!(
        case["norm_probe_is_independent_of_main_recurrence"].as_bool() == Some(true),
        "dense oracle norm probe scope"
    );
    let stages = &case["stages"];
    ensure!(
        stages.as_object().is_some_and(|v| v.len() == 12),
        "dense oracle stage count"
    );
    for name in ["q_bf16", "k_bf16"] {
        descriptor_guard(&stages[name], &[1, m, HK, D], "bfloat16")?;
    }
    for name in ["v_bf16", "raw_y_bf16"] {
        descriptor_guard(&stages[name], &[1, m, HV, D], "bfloat16")?;
    }
    descriptor_guard(&stages["conv_history_bf16"], &[1, 3, C], "bfloat16")?;
    for name in ["decay_f32", "beta_f32"] {
        descriptor_guard(&stages[name], &[1, m, HV], "float32")?;
    }
    descriptor_guard(&stages["beta_bf16"], &[1, m, HV], "bfloat16")?;
    for name in ["conv_raw_bf16", "conv_activated_bf16"] {
        descriptor_guard(&stages[name], &[1, m, C], "bfloat16")?;
    }
    descriptor_guard(&stages["state_f32"], &[1, HV, D, D], "float32")?;
    descriptor_guard(&stages["norm_bf16"], &[1, m, HV * D], "bfloat16")?;
    let probes = &case["probes"];
    ensure!(
        probes.as_object().is_some_and(|v| v.len() == 3),
        "dense oracle probe count"
    );
    for name in ["q_bf16", "k_bf16"] {
        descriptor_guard(&probes[name], &[1, m, HK, D], "bfloat16")?;
    }
    descriptor_guard(&probes["norm_bf16"], &[1, m, HV * D], "bfloat16")?;
    let composed = &case["composed_conv_crosscheck"];
    ensure!(
        text(composed, "scope")?
            == "independent composed production reference; never replaces untouched served prework",
        "dense oracle composed scope"
    );
    for name in ["conv_raw_bf16", "conv_activated_bf16"] {
        descriptor_guard(&composed[name], &[1, m, C], "bfloat16")?;
    }
    let counts = &composed["mismatch_counts"];
    ensure!(
        counts.as_object().is_some_and(|v| v.len() == 5),
        "dense oracle composed count fields"
    );
    for (name, limit) in [
        ("conv_raw", m * C),
        ("conv_activated", m * C),
        ("q", m * HK * D),
        ("k", m * HK * D),
        ("v", m * HV * D),
    ] {
        ensure!(
            counts[name].as_u64().is_some_and(|n| n <= limit as u64),
            "dense oracle composed mismatch count"
        );
    }
    Ok(())
}

struct Oracle {
    receipt: Value,
    source_sha: String,
    parent: PathBuf,
}

fn load_oracle() -> Result<Oracle> {
    hash_is(include_bytes!("metal/gdn.metal"), GDN_METAL_SHA)?;
    hash_is(include_bytes!("metal/official_gdn_pipelined.metal"), TB12_SHA)?;
    let path = PathBuf::from(
        std::env::var_os("STARK_QWEN38_GDN_DENSE_ORACLE")
            .ok_or_else(|| anyhow::anyhow!("dense oracle receipt required"))?,
    );
    let meta = std::fs::metadata(&path)
        .map_err(|_| anyhow::anyhow!("dense oracle receipt metadata"))?;
    ensure!(meta.is_file() && meta.len() <= 131072, "dense oracle receipt bounds");
    let bytes = std::fs::read(&path)
        .map_err(|_| anyhow::anyhow!("dense oracle receipt read"))?;
    let receipt: Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("dense oracle receipt JSON"))?;
    for (key, expected) in [
        ("schema", "stark-private-stock-dense-gdn-prefill-oracle-v2"),
        ("status", "complete"),
        ("controller_sha256", CONTROLLER_SHA),
        ("fixture_spec_sha256", SPEC_SHA),
        (
            "pipelined_helper_sha256",
            "bee45491e95697dbb791b83f340b1adaff191e15a4b2ab15ccaf104ac242f653",
        ),
        (
            "stock_source_sha256",
            "4de16729aa7614a55215acfa14c7df69094ba52b4018b138c1c5ff6b4333d86a",
        ),
        (
            "transformer_source_sha256",
            "b2879bdd3f15bef2607df9aeb205977468fccde0e5f3d9a2ba1010384fe53489",
        ),
        (
            "stock_prework_source_sha256",
            "ad9f85bc5a473a45d749e804630e2f92f993eec2e6ee3ad450c68b4631ef4d06",
        ),
        (
            "diagnostic_tap_source_sha256",
            "490bf7b2bc4cfbb0d7405643b89e140ab238545bfa50bb663fbc9459282e470e",
        ),
        (
            "stock_norm_source_sha256",
            "1fb99af7f1391ee45422650fa88b99800baabce10655f49767817700599dff30",
        ),
        (
            "compiler_source_sha256",
            "d2254ea5cb2ac282f65c2869c6ef56decc97669e2a5fc26fba9d831b60569c9f",
        ),
        ("server_revision", "046b647686419df06c6c7132fb1d15d76185c3e5"),
        ("math_mode", "Safe: pinned MLX default, no environment overrides"),
        ("msl_version", "4.0: pinned macOS26 selection"),
        ("adversarial_negative_bf16_case", "c0db"),
        ("state_layout", "ValueKey"),
        ("scope", SCOPE),
    ] {
        ensure!(text(&receipt, key)? == expected, "dense oracle provenance field");
    }
    for (key, expected) in [
        ("synthetic_only", true),
        ("model_loaded", false),
        ("raw_arrays_persisted", false),
        ("diagnostic_only", true),
        ("full_model_equivalence_claimed", false),
        ("native_precision_inferred_from_dtype", false),
    ] {
        ensure!(
            receipt[key].as_bool() == Some(expected),
            "dense oracle execution flags"
        );
    }
    ensure!(
        receipt["TAB"].as_u64() == Some(1)
            && receipt["adversarial_bf16_case_count"].as_u64() == Some(33),
        "dense oracle TAB/adversarial coverage"
    );
    ensure!(
        text(&receipt, "macos_version")?.starts_with("26."),
        "dense oracle stock compiler host"
    );
    let assets = &receipt["stock_asset_sha256"];
    ensure!(
        assets.as_object().is_some_and(|v| v.len() == 4),
        "dense oracle asset count"
    );
    for (name, expected) in [
        (
            "lib/libmlx.dylib",
            "fc504cec0f86f4a20b111d60dd1112df178bfd7d9b7978d4d735fa1a9c9bf8d4",
        ),
        (
            "lib/libmlxc.dylib",
            "83ebedc48c226376af608b939e17256550d4baa1ad0cc276195c676fdc55c501",
        ),
        (
            "lib/libjaccl.dylib",
            "26c636f17b57062978ac100fe6b1b146d6d9891c1ab1935fb4db680c6fbb3556",
        ),
        (
            "lib/mlx.metallib",
            "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1",
        ),
    ] {
        ensure!(text(assets, name)? == expected, "dense oracle asset provenance");
    }
    let cases = receipt["fixtures"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("dense oracle fixtures"))?;
    ensure!(cases.len() == ROWS.len(), "dense oracle fixture count");
    for (index, rows) in ROWS.into_iter().enumerate() {
        case_guards(&cases[index], &fixture(rows))?;
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("dense oracle receipt parent"))?
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("dense oracle receipt parent"))?;
    Ok(Oracle { receipt, source_sha: digest(&bytes), parent })
}

fn native_history(stock: &[f32]) -> Vec<f32> {
    let mut native = vec![0.0; 3 * C];
    for ch in 0..C {
        for tap in 0..3 {
            native[ch * 3 + tap] = stock[tap * C + ch];
        }
    }
    native
}
fn stock_history(native: &Tensor) -> Result<Vec<u8>> {
    let values = native.to_f32()?;
    let mut stock = vec![0.0; 3 * C];
    for ch in 0..C {
        for tap in 0..3 {
            stock[tap * C + ch] = values[ch * 3 + tap];
        }
    }
    Ok(bf_bytes(&stock))
}
fn separated_qk(tensor: &Tensor, rows: usize) -> (Vec<u8>, Vec<u8>) {
    let bytes = tensor.raw_bytes();
    let width = HK * D * 2;
    let mut q = Vec::with_capacity(rows * width);
    let mut k = Vec::with_capacity(rows * width);
    for row in 0..rows {
        q.extend_from_slice(&bytes[row * 2 * width..row * 2 * width + width]);
        k.extend_from_slice(&bytes[row * 2 * width + width..(row + 1) * 2 * width]);
    }
    (q, k)
}
fn finite(bytes: &[u8], dtype: &str) -> bool {
    if dtype == "bfloat16" {
        bytes.chunks_exact(2).all(|v| {
            half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32().is_finite()
        })
    } else {
        bytes
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]).is_finite())
    }
}
fn observed(bytes: &[u8], expected: &Value) -> Value {
    let actual = digest(bytes);
    json!({"available": true, "sha256": actual, "bytes": bytes.len(), "shape": expected["shape"],
        "dtype": expected["dtype"], "expected_sha256": expected["sha256"],
        "matches_stock": Some(actual.as_str()) == expected["sha256"].as_str(),
        "all_finite": finite(bytes, expected["dtype"].as_str().unwrap_or("invalid"))})
}

// Runs all three native legs before any numerical comparison can fail.
fn collect_once(
    ctx: &MetalContext,
    fixture: &Fixture,
    expected: &Value,
) -> Result<Value> {
    let m = fixture.rows;
    let x = Tensor::from_f32_as_bf16(ctx, &fixture.qkv, &[m, C])?;
    let history_in =
        Tensor::from_f32_as_bf16(ctx, &native_history(&fixture.history), &[C, 3])?;
    let history_out = Tensor::zeros(ctx, &[C, 3], DType::BF16)?;
    let mut native_weights = Vec::with_capacity(4 * C);
    for tap in 0..4 {
        for ch in 0..C {
            native_weights.push(fixture.weights[ch * 4 + tap]);
        }
    }
    let weights = Tensor::from_f32_as_bf16(ctx, &native_weights, &[4, C])?;
    let conv = Tensor::zeros(ctx, &[m, C], DType::BF16)?;
    let a = Tensor::from_f32_as_bf16(ctx, &fixture.a, &[m, HV])?;
    let b = Tensor::from_f32_as_bf16(ctx, &fixture.b, &[m, HV])?;
    let a_log = Tensor::from_f32(ctx, &fixture.a_log, &[HV])?;
    let dt = Tensor::from_f32_as_bf16(ctx, &fixture.dt, &[HV])?;
    let qk = Tensor::zeros(ctx, &[m, 2 * HK * D], DType::BF16)?;
    let g = Tensor::zeros(ctx, &[m, HV], DType::F32)?;
    let beta = Tensor::zeros(ctx, &[m, HV], DType::F32)?;
    let staging =
        GdnRegscanStaging { qk_norm: &qk, decay: &g, beta: &beta, chunk: None };
    let state = Tensor::from_f32(ctx, &fixture.state, &[HV, D, D])?;
    let raw = Tensor::zeros(ctx, &[m, HV, D], DType::BF16)?;
    let z = Tensor::from_f32_as_bf16(ctx, &fixture.z, &[m, HV * D])?;
    // The same BF16 gamma values are widened exactly for the native F32 API.
    let gamma = Tensor::from_f32(ctx, &fixture.gamma, &[D])?;
    let norm = Tensor::zeros(ctx, &[m, HV * D], DType::BF16)?;
    let probe_conv = Tensor::from_f32_as_bf16(ctx, &fixture.probe_conv, &[m, C])?;
    let probe_qk = Tensor::zeros(ctx, &[m, 2 * HK * D], DType::BF16)?;
    let probe_y = Tensor::from_f32_as_bf16(ctx, &fixture.probe_y, &[m, HV, D])?;
    let probe_norm = Tensor::zeros(ctx, &[m, HV * D], DType::BF16)?;
    ensure!(
        value_key_pipelined_prefill_selected(true, GdnStateLayout::ValueKey, m, false),
        "dense diagnostic TB12 selector"
    );
    ensure!(
        pipelined_prefill_buffers_aligned(&conv, &staging, &state, &raw),
        "dense diagnostic TB12 alignment"
    );
    let pass = ctx.begin_concurrent()?;
    conv1d_prefill_with_rounding(
        ctx,
        &pass,
        &history_in,
        &history_out,
        &x,
        &weights,
        &conv,
        true,
    )?;
    pass.level_barrier(&[&history_out, &conv])?;
    gdn_prefill_mid_with_layout(
        ctx,
        &pass,
        &conv,
        &a,
        &b,
        &a_log,
        &dt,
        &staging,
        &state,
        &raw,
        1.0f32 / 128.0f32.sqrt(),
        HK,
        None,
        true,
        GdnStateLayout::ValueKey,
    )?;
    pass.level_barrier(&[&qk, &g, &beta, &raw, &state])?;
    gated_rmsnorm_with_rounding(
        ctx,
        &pass,
        &raw,
        &z,
        &gamma,
        &norm,
        1e-6f32,
        GdnGate::Sigmoid,
        true,
    )?;
    gdn_qk_rms_mlx(
        ctx,
        &pass,
        &probe_conv,
        &probe_qk,
        1.0f32 / 128.0f32.sqrt(),
        HK,
        HV,
    )?;
    gated_rmsnorm_with_rounding(
        ctx,
        &pass,
        &probe_y,
        &z,
        &gamma,
        &probe_norm,
        1e-6f32,
        GdnGate::Sigmoid,
        true,
    )?;
    pass.commit()?.wait()?;
    let (q, k) = separated_qk(&qk, m);
    let (probe_q, probe_k) = separated_qk(&probe_qk, m);
    let history = stock_history(&history_out)?;
    let mut v = Vec::with_capacity(m * HV * D * 2);
    for row in 0..m {
        let start = row * C * 2;
        v.extend_from_slice(&conv.raw_bytes()[start + 2 * HK * D * 2..start + C * 2]);
    }
    let beta_bf16 = bf_bytes(&beta.to_f32()?);
    let expected_stages = &expected["stages"];
    let mut stages = serde_json::Map::new();
    stages.insert("conv_raw_bf16".into(), json!({"available": false, "reason": "production native conv exposes only activated BF16 output; raw conv was not instrumented", "stock_sha256": expected_stages["conv_raw_bf16"]["sha256"]}));
    for (name, bytes) in [
        ("conv_activated_bf16", conv.raw_bytes()),
        ("conv_history_bf16", history.as_slice()),
        ("q_bf16", q.as_slice()),
        ("k_bf16", k.as_slice()),
        ("v_bf16", v.as_slice()),
        ("decay_f32", g.raw_bytes()),
        ("beta_bf16", beta_bf16.as_slice()),
        ("beta_f32", beta.raw_bytes()),
        ("raw_y_bf16", raw.raw_bytes()),
        ("state_f32", state.raw_bytes()),
        ("norm_bf16", norm.raw_bytes()),
    ] {
        stages.insert(name.into(), observed(bytes, &expected_stages[name]));
    }
    let expected_probes = &expected["probes"];
    let probes = json!({"q_bf16": observed(&probe_q, &expected_probes["q_bf16"]),
        "k_bf16": observed(&probe_k, &expected_probes["k_bf16"]),
        "norm_bf16": observed(probe_norm.raw_bytes(), &expected_probes["norm_bf16"])});
    let prework_matches = [
        "conv_activated_bf16",
        "conv_history_bf16",
        "q_bf16",
        "k_bf16",
        "v_bf16",
        "decay_f32",
        "beta_bf16",
        "beta_f32",
    ]
    .iter()
    .all(|name| stages[*name]["matches_stock"].as_bool() == Some(true));
    let raw_matches = stages["raw_y_bf16"]["matches_stock"].as_bool() == Some(true);
    Ok(json!({"rows": m, "salt": fixture.salt, "stages": stages, "probes": probes,
        "recurrence_comparison_interpretable": prework_matches,
        "recurrence_comparison_requires_matching_prework_inputs": true,
        "main_norm_comparison_interpretable": raw_matches,
        "norm_probe_is_independent_of_main_recurrence": true,
        "rms_probe_has_identical_reconstructed_input": true,
        "inputs": fixture.input_hashes(),
        "stock_composed_conv_crosscheck": expected["composed_conv_crosscheck"]}))
}

fn comparable_exact(run: &Value) -> bool {
    ["stages", "probes"].into_iter().all(|group| {
        run[group].as_object().is_some_and(|values| {
            values.values().all(|stage| {
                stage["available"].as_bool() == Some(false)
                    || (stage["matches_stock"].as_bool() == Some(true)
                        && stage["all_finite"].as_bool() == Some(true))
            })
        })
    })
}
fn repeats(first: &Value, second: &Value) -> (Value, bool) {
    let mut report = serde_json::Map::new();
    let mut all = true;
    for group in ["stages", "probes"] {
        let mut stages = serde_json::Map::new();
        if let Some(values) = first[group].as_object() {
            for (name, first_stage) in values {
                if first_stage["available"].as_bool() == Some(false) {
                    continue;
                }
                let exact = first_stage["sha256"] == second[group][name]["sha256"];
                all &= exact;
                stages.insert(name.clone(), json!({"exact": exact, "repeat_sha256": second[group][name]["sha256"]}));
            }
        } else {
            all = false;
        }
        report.insert(group.into(), Value::Object(stages));
    }
    (Value::Object(report), all)
}

fn collect(oracle: &Oracle) -> Result<(Value, bool)> {
    let ctx = MetalContext::new()?;
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_prefill_pipelined_value_key_mlx",
        PIPELINED_PREFILL_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    ensure!(pipeline.thread_execution_width() == 32, "dense diagnostic TB12 SIMD32");
    let mut cases = Vec::new();
    let mut all_exact = true;
    for (index, rows) in ROWS.into_iter().enumerate() {
        let fixture = fixture(rows);
        let expected = &oracle.receipt["fixtures"][index];
        let mut first = collect_once(&ctx, &fixture, expected)?;
        let second = collect_once(&ctx, &fixture, expected)?;
        let (repeat, repeat_exact) = repeats(&first, &second);
        let exact =
            comparable_exact(&first) && comparable_exact(&second) && repeat_exact;
        all_exact &= exact;
        first["repeat"] = repeat;
        first["native_repeats_exact"] = Value::from(repeat_exact);
        first["all_comparable_stages_exact"] = Value::from(exact);
        cases.push(first);
    }
    Ok((
        json!({"fixtures": cases, "native_tb12_simd_width": pipeline.thread_execution_width(),
        "native_tb12_static_threadgroup_memory_bytes": pipeline.static_threadgroup_memory_length(),
        "all_comparable_stages_exact": all_exact}),
        all_exact,
    ))
}

fn output_path(oracle: &Oracle) -> Result<PathBuf> {
    let path = PathBuf::from(
        std::env::var_os("STARK_QWEN38_GDN_DENSE_DIAGNOSTIC_OUTPUT")
            .ok_or_else(|| anyhow::anyhow!("dense diagnostic output path required"))?,
    );
    ensure!(
        path.is_absolute() && !path.exists(),
        "dense diagnostic output must be fresh absolute path"
    );
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("dense diagnostic output parent"))?
        .canonicalize()
        .map_err(|_| anyhow::anyhow!("dense diagnostic output parent"))?;
    ensure!(
        parent == oracle.parent && path.file_name().is_some(),
        "dense diagnostic output must share private receipt directory"
    );
    Ok(path)
}
fn atomic_private_json(path: &Path, value: &Value) -> Result<()> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| anyhow::anyhow!("dense diagnostic output clock"))?
        .as_nanos();
    let temporary = path.with_file_name(format!(
        ".dense-gdn-diagnostic-{}-{nonce}.tmp",
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|_| anyhow::anyhow!("dense diagnostic JSON encoding"))?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)
        .map_err(|_| anyhow::anyhow!("dense diagnostic private temporary creation"))?;
    let stored = file
        .write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all());
    drop(file);
    if stored.is_err() {
        let _ = std::fs::remove_file(&temporary);
        return Err(anyhow::anyhow!("dense diagnostic private write failed"));
    }
    // Atomic publication without replacing an existing destination. A hard
    // link keeps the completed 0600 file, then its temporary name is removed.
    let published = std::fs::hard_link(&temporary, path);
    let _ = std::fs::remove_file(&temporary);
    ensure!(published.is_ok(), "dense diagnostic atomic publication failed");
    Ok(())
}

#[test]
#[ignore = "requires completed private dense stock oracle and physical Metal; mismatch is diagnostic"]
fn official_dense_prefill_collects_all_stage_hashes_before_numerical_failure()
-> Result<()> {
    let oracle = load_oracle()?;
    let output = output_path(&oracle)?;
    let mut report = json!({"schema": "stark-private-native-dense-gdn-diagnostic-v1",
        "synthetic_only": true, "model_loaded": false, "raw_arrays_persisted": false,
        "diagnostic_only": true, "full_model_equivalence_claimed": false,
        "native_precision_inferred_from_dtype": false, "speculation": false, "scope": SCOPE,
        "oracle_receipt_sha256": oracle.source_sha, "oracle_controller_sha256": CONTROLLER_SHA,
        "fixture_spec_sha256": SPEC_SHA,
        "source_sha256": {"native_gdn_metal": digest(include_bytes!("metal/gdn.metal")), "native_tb12_metal": digest(include_bytes!("metal/official_gdn_pipelined.metal")),
            "native_gdn_rust": digest(include_bytes!("gdn.rs")),
            "diagnostic_test_rust": digest(include_bytes!("official_gdn_dense_diagnostic_tests.rs"))},
        "native_precision_policy": "BF16 activations and convolution; F32 recurrence and decay; beta BF16 then exact F32 widening; BF16 gamma exactly widened to F32; original prework compiler policy, scoped Safe TB12 MSL3.1 and scoped norm pragmas",
        "missing_native_stage": "raw convolution before SiLU is unavailable without instrumenting production; activated convolution is compared",
        "coverage": {"available_main_stages": ["conv_activated_bf16", "conv_history_bf16", "q_bf16", "k_bf16", "v_bf16", "decay_f32", "beta_bf16", "beta_f32", "raw_y_bf16", "state_f32", "norm_bf16"],
            "unavailable_main_stages": ["conv_raw_bf16"], "independent_probes": ["q_bf16", "k_bf16", "norm_bf16"],
            "raw_convolution_unavailability_counts_as_mismatch": false},
        "stock_source_provenance": {"stock_source_sha256": oracle.receipt["stock_source_sha256"],
            "transformer_source_sha256": oracle.receipt["transformer_source_sha256"],
            "stock_prework_source_sha256": oracle.receipt["stock_prework_source_sha256"],
            "diagnostic_tap_source_sha256": oracle.receipt["diagnostic_tap_source_sha256"],
            "stock_norm_source_sha256": oracle.receipt["stock_norm_source_sha256"],
            "compiler_source_sha256": oracle.receipt["compiler_source_sha256"]}});
    let (data, all_exact) = match collect(&oracle) {
        Ok(result) => result,
        Err(_) => {
            report["status"] = Value::from("incomplete");
            report["failure_category"] =
                Value::from("native_gpu_or_stage_execution_failed");
            atomic_private_json(&output, &report)?;
            return Err(anyhow::anyhow!(
                "dense native diagnostic execution failed; fixed-category private report written"
            ));
        }
    };
    report["status"] = Value::from(if all_exact {
        "complete_exact"
    } else {
        "complete_with_mismatches"
    });
    report["native"] = data;
    atomic_private_json(&output, &report)?;
    ensure!(
        all_exact,
        "dense native stage mismatch; complete private aggregate diagnostic written"
    );
    Ok(())
}
