// SPDX-License-Identifier: Apache-2.0
// Private synthetic aggregate/hash controls for the pinned stock TB12 kernel.
// No model, prompt, token vector, raw tensor or vocabulary data is logged.
use serde_json::Value;

use super::*;
use crate::sha256::{hex, sha256};

const HK: usize = 16;
const HV: usize = 48;
const D: usize = 128;
const C: usize = 10240;
const ROWS: [usize; 7] = [11, 12, 13, 63, 64, 65, 129];
const CONTROLLER_SHA: &str =
    "bee45491e95697dbb791b83f340b1adaff191e15a4b2ab15ccaf104ac242f653";
const STOCK_SHA: &str =
    "4de16729aa7614a55215acfa14c7df69094ba52b4018b138c1c5ff6b4333d86a";
const NATIVE_SHA: &str =
    "2c099d5cb38b0ba3dd819762554f0fa9ce91e8709dd5b3815d9423d2b73e5313";
const SPEC: &str = "stark-stock-gdn-tb12-fixture-v1;q=bf(i*7+3%127-63)/4096;k=bf(i*11+5%113-56)/256;v=bf(i*13+7%109-54)/53;g=f32(.98+(i%17-8)/2048);beta=bf((i%15+1)/16);state=dense17-257/997+11/65537;firstzero";

fn bf(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
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

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("pipelined oracle string field"))
}

fn hash_is(bytes: &[u8], expected: &str, category: &str) -> Result<()> {
    let actual = hex(&sha256(bytes));
    ensure!(
        actual == expected,
        "{category}: actual_sha256={actual}, expected_sha256={expected}"
    );
    Ok(())
}

fn input_is(case: &Value, name: &str, bytes: &[u8]) -> Result<()> {
    hash_is(bytes, text(&case["inputs"], name)?, "pipelined input fixture")
}

fn described_is(
    bytes: &[u8],
    descriptor: &Value,
    shape: &[usize],
    dtype: &str,
    category: &str,
) -> Result<()> {
    let expected_shape =
        shape.iter().map(|n| Value::from(*n as u64)).collect::<Vec<_>>();
    ensure!(
        descriptor["shape"].as_array() == Some(&expected_shape),
        "{category}: descriptor shape"
    );
    ensure!(
        descriptor["dtype"].as_str() == Some(dtype),
        "{category}: descriptor dtype"
    );
    ensure!(
        descriptor["bytes"].as_u64() == Some(bytes.len() as u64),
        "{category}: descriptor byte count"
    );
    hash_is(bytes, text(descriptor, "sha256")?, category)
}

fn oracle() -> Result<Value> {
    hash_is(
        include_bytes!("metal/official_gdn_pipelined.metal"),
        NATIVE_SHA,
        "native recurrence source",
    )?;
    let path =
        std::env::var_os("STARK_QWEN38_GDN_PIPELINED_ORACLE").ok_or_else(|| {
            anyhow::anyhow!("STARK_QWEN38_GDN_PIPELINED_ORACLE receipt required")
        })?;
    let metadata = std::fs::metadata(&path)
        .map_err(|_| anyhow::anyhow!("pipelined oracle receipt metadata"))?;
    ensure!(
        metadata.is_file() && metadata.len() <= 131072,
        "pipelined oracle receipt bounds"
    );
    let bytes = std::fs::read(&path)
        .map_err(|_| anyhow::anyhow!("pipelined oracle receipt read"))?;
    let receipt: Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("pipelined oracle receipt JSON"))?;
    for (key, expected) in [
        ("schema", "stark-private-stock-gdn-pipelined-oracle-v1"),
        ("status", "complete"),
        ("controller_sha256", CONTROLLER_SHA),
        (
            "k1_helper_sha256",
            "fb5b3d66487ce94a9a6110ef1e199c536a1c6d0bd2b8be681da90fc25226553f",
        ),
        ("stock_source_sha256", STOCK_SHA),
        (
            "transformer_source_sha256",
            "b2879bdd3f15bef2607df9aeb205977468fccde0e5f3d9a2ba1010384fe53489",
        ),
        ("native_shader_sha256", NATIVE_SHA),
        ("server_revision", "046b647686419df06c6c7132fb1d15d76185c3e5"),
        (
            "license_sha256",
            "dfe22e0f100fe7f2c2808f1af6f488e7339cf8deb7504153602b4169ed7d90b4",
        ),
        ("state_layout", "ValueKey"),
        ("state_dtype", "float32"),
        ("activation_dtype", "bfloat16"),
        ("native_math_mode_candidate", "Safe"),
        ("native_msl_candidate", "3.1"),
    ] {
        ensure!(text(&receipt, key)? == expected, "pipelined oracle provenance field");
    }
    hash_is(
        SPEC.as_bytes(),
        text(&receipt, "fixture_spec_sha256")?,
        "pipelined fixture specification",
    )?;
    for (key, expected) in [
        ("synthetic_only", true),
        ("model_loaded", false),
        ("raw_arrays_persisted", false),
        ("full_model_equivalence_claimed", false),
    ] {
        ensure!(
            receipt[key].as_bool() == Some(expected),
            "pipelined oracle execution flags"
        );
    }
    ensure!(
        receipt["stock_source_lines"].as_u64() == Some(609),
        "pipelined oracle stock source length"
    );
    ensure!(
        receipt["new_global_scratch_bytes"].as_u64() == Some(0),
        "pipelined oracle scratch policy"
    );
    let assets = &receipt["stock_asset_sha256"];
    ensure!(
        assets.as_object().is_some_and(|m| m.len() == 4),
        "pipelined oracle stock asset count"
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
        ensure!(
            text(assets, name)? == expected,
            "pipelined oracle stock asset provenance"
        );
    }
    let fixtures = receipt["fixtures"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("pipelined oracle fixtures"))?;
    ensure!(fixtures.len() == 8, "pipelined oracle fixture count");
    for (index, rows) in ROWS.into_iter().enumerate() {
        let case = &fixtures[index];
        ensure!(
            text(case, "fixture")? == "direct_staged",
            "pipelined direct fixture kind"
        );
        ensure!(
            case["rows"].as_u64() == Some(rows as u64),
            "pipelined direct fixture rows/order"
        );
        ensure!(
            case["dense_initial_state"].as_bool() == Some(index != 0),
            "pipelined direct initial state"
        );
        ensure!(
            case["inputs"].as_object().is_some_and(|m| m.len() == 9),
            "pipelined direct input count"
        );
        ensure!(
            case["repeat_mismatch_counts"] == serde_json::json!([0, 0]),
            "pipelined stock recurrence repeat"
        );
        direct_fixture(rows, index != 0).check_inputs(case)?;
    }
    let sparse = &fixtures[7];
    ensure!(
        text(sparse, "fixture")? == "sparse_prework_recurrence_norm",
        "pipelined sparse fixture kind"
    );
    ensure!(
        sparse["rows"].as_u64() == Some(65) && sparse["TAB"].as_u64() == Some(1),
        "pipelined sparse fixture route"
    );
    ensure!(
        sparse["inputs"].as_object().is_some_and(|m| m.len() == 8),
        "pipelined sparse input count"
    );
    ensure!(
        sparse["repeat_mismatch_counts"]
            == serde_json::json!({"prework": 0, "raw_y": 0, "state_out": 0, "norm_out": 0}),
        "pipelined stock sparse repeat"
    );
    ensure!(
        text(sparse, "scope")?
            == "sparse bounded control; does not qualify dense prework/norm reduction policy",
        "pipelined sparse qualification scope"
    );
    sparse_fixture().check_inputs(sparse)?;
    Ok(receipt)
}

struct DirectFixture {
    rows: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
    qk: Vec<f32>,
    qkv: Vec<f32>,
}

impl DirectFixture {
    fn check_inputs(&self, receipt: &Value) -> Result<()> {
        for (name, values) in [
            ("q_bf16_sha256", &self.q),
            ("k_bf16_sha256", &self.k),
            ("v_bf16_sha256", &self.v),
            ("beta_bf16_sha256", &self.beta),
            ("qk_bf16_sha256", &self.qk),
            ("qkv_bf16_sha256", &self.qkv),
        ] {
            input_is(receipt, name, &bf_bytes(values))?;
        }
        for (name, values) in [
            ("g_f32_sha256", &self.g),
            ("beta_f32_sha256", &self.beta),
            ("state_f32_sha256", &self.state),
        ] {
            input_is(receipt, name, &f32_bytes(values))?;
        }
        Ok(())
    }
}

fn direct_fixture(rows: usize, dense: bool) -> DirectFixture {
    let q: Vec<_> = (0..rows * HK * D)
        .map(|i| bf((((i * 7 + 3) % 127) as i32 - 63) as f32 / 4096.0f32))
        .collect();
    let k: Vec<_> = (0..rows * HK * D)
        .map(|i| bf((((i * 11 + 5) % 113) as i32 - 56) as f32 / 256.0f32))
        .collect();
    let v: Vec<_> = (0..rows * HV * D)
        .map(|i| bf((((i * 13 + 7) % 109) as i32 - 54) as f32 / 53.0f32))
        .collect();
    let g: Vec<_> = (0..rows * HV)
        .map(|i| 0.98f32 + (i as i32 % 17 - 8) as f32 / 2048.0f32)
        .collect();
    let beta: Vec<_> =
        (0..rows * HV).map(|i| bf((i % 15 + 1) as f32 / 16.0f32)).collect();
    let state = (0..HV * D * D)
        .map(|i| {
            if dense {
                let first = (((i * 17 + 3) % 257) as i32 - 128) as f32 / 997.0f32;
                let second = (i as i32 % 11 - 5) as f32 / 65537.0f32;
                first + second
            } else {
                0.0
            }
        })
        .collect();
    let mut qk = Vec::with_capacity(rows * 2 * HK * D);
    let mut qkv = Vec::with_capacity(rows * C);
    for row in 0..rows {
        qk.extend_from_slice(&q[row * HK * D..(row + 1) * HK * D]);
        qk.extend_from_slice(&k[row * HK * D..(row + 1) * HK * D]);
        qkv.extend(std::iter::repeat_n(bf(-7.25), 2 * HK * D));
        qkv.extend_from_slice(&v[row * HV * D..(row + 1) * HV * D]);
    }
    DirectFixture { rows, q, k, v, g, beta, state, qk, qkv }
}

fn pipeline_metadata(ctx: &MetalContext) -> Result<()> {
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_prefill_pipelined_value_key_mlx",
        PIPELINED_PREFILL_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    ensure!(pipeline.thread_execution_width() == 32, "pipelined native SIMD width");
    // Nominal array storage is 7392 bytes; reflected allocation may differ
    // after compiler lifetime packing. This is only a bounded metadata check.
    ensure!(
        (1..=32768).contains(&pipeline.static_threadgroup_memory_length()),
        "pipelined native threadgroup memory"
    );
    Ok(())
}

fn direct_once(
    ctx: &MetalContext,
    fixture: &DirectFixture,
    receipt: &Value,
) -> Result<(String, String)> {
    let rows = fixture.rows;
    let qkv = Tensor::from_f32_as_bf16(ctx, &fixture.qkv, &[rows, C])?;
    let qk = Tensor::from_f32_as_bf16(ctx, &fixture.qk, &[rows, 2 * HK * D])?;
    let decay = Tensor::from_f32(ctx, &fixture.g, &[rows, HV])?;
    let beta = Tensor::from_f32(ctx, &fixture.beta, &[rows, HV])?;
    let state = Tensor::from_f32(ctx, &fixture.state, &[HV, D, D])?;
    let out = Tensor::zeros(ctx, &[rows, HV, D], DType::BF16)?;
    let staging =
        GdnRegscanStaging { qk_norm: &qk, decay: &decay, beta: &beta, chunk: None };
    let pass = ctx.begin_concurrent()?;
    // Deliberately bypass only the >=64 production selector, exercising TB12
    // tail handling at 11/12/13 with the production helper and shader.
    gdn_prefill_pipelined_value_key_mlx(
        ctx,
        &pass,
        &qkv,
        &staging,
        &state,
        &out,
        GdnStateLayout::ValueKey,
    )?;
    pass.commit()?.wait()?;
    described_is(
        out.raw_bytes(),
        &receipt["raw_y"],
        &[1, rows, HV, D],
        "bfloat16",
        "direct native raw Y",
    )?;
    described_is(
        state.raw_bytes(),
        &receipt["state_out"],
        &[1, HV, D, D],
        "float32",
        "direct native F32 state",
    )?;
    ensure!(
        state.to_f32()?.iter().all(|x| x.is_finite()),
        "direct native state finite"
    );
    input_is(receipt, "qkv_bf16_sha256", qkv.raw_bytes())?;
    input_is(receipt, "qk_bf16_sha256", qk.raw_bytes())?;
    input_is(receipt, "g_f32_sha256", decay.raw_bytes())?;
    input_is(receipt, "beta_f32_sha256", beta.raw_bytes())?;
    Ok((hex(&sha256(out.raw_bytes())), hex(&sha256(state.raw_bytes()))))
}

struct Guarded {
    owner: Tensor,
    view: Tensor,
    before: Vec<u8>,
    prefix_bytes: usize,
    payload_bytes: usize,
}

impl Guarded {
    fn new(
        ctx: &MetalContext,
        payload: &[u8],
        shape: &[usize],
        dtype: DType,
        prefix_elements: usize,
    ) -> Result<Self> {
        let word = if dtype == DType::BF16 {
            0x3e55u16.to_le_bytes().to_vec()
        } else {
            0x3eaaaaabu32.to_le_bytes().to_vec()
        };
        let prefix_bytes = prefix_elements * word.len();
        let mut before = word.repeat(prefix_elements);
        before.extend_from_slice(payload);
        before.extend_from_slice(&word.repeat(8));
        let owner =
            Tensor::from_bytes(ctx, &before, &[before.len() / word.len()], dtype)?;
        let view = owner.view(prefix_elements, shape)?;
        Ok(Self { owner, view, before, prefix_bytes, payload_bytes: payload.len() })
    }

    fn check(&self, read_only: bool) -> Result<()> {
        let actual = self.owner.raw_bytes();
        let end = self.prefix_bytes + self.payload_bytes;
        ensure!(
            actual[..self.prefix_bytes] == self.before[..self.prefix_bytes]
                && actual[end..] == self.before[end..],
            "pipelined aligned view canary mismatch"
        );
        ensure!(
            !read_only || actual == self.before.as_slice(),
            "pipelined read-only input changed"
        );
        Ok(())
    }
}

fn aligned_views_and_refusals(
    ctx: &MetalContext,
    fixture: &DirectFixture,
    receipt: &Value,
) -> Result<()> {
    let rows = fixture.rows;
    let qkv = Guarded::new(ctx, &bf_bytes(&fixture.qkv), &[rows, C], DType::BF16, 4)?;
    let qk =
        Guarded::new(ctx, &bf_bytes(&fixture.qk), &[rows, 2 * HK * D], DType::BF16, 4)?;
    let decay = Guarded::new(ctx, &f32_bytes(&fixture.g), &[rows, HV], DType::F32, 4)?;
    let beta =
        Guarded::new(ctx, &f32_bytes(&fixture.beta), &[rows, HV], DType::F32, 4)?;
    let state =
        Guarded::new(ctx, &f32_bytes(&fixture.state), &[HV, D, D], DType::F32, 4)?;
    let out = Guarded::new(
        ctx,
        &vec![0u8; rows * HV * D * 2],
        &[rows, HV, D],
        DType::BF16,
        4,
    )?;
    let staging = GdnRegscanStaging {
        qk_norm: &qk.view,
        decay: &decay.view,
        beta: &beta.view,
        chunk: None,
    };
    ensure!(
        pipelined_prefill_buffers_aligned(&qkv.view, &staging, &state.view, &out.view),
        "aligned fixture rejected"
    );
    let pass = ctx.begin_concurrent()?;
    gdn_prefill_pipelined_value_key_mlx(
        ctx,
        &pass,
        &qkv.view,
        &staging,
        &state.view,
        &out.view,
        GdnStateLayout::ValueKey,
    )?;
    pass.commit()?.wait()?;
    described_is(
        out.view.raw_bytes(),
        &receipt["raw_y"],
        &[1, rows, HV, D],
        "bfloat16",
        "aligned native raw Y",
    )?;
    described_is(
        state.view.raw_bytes(),
        &receipt["state_out"],
        &[1, HV, D, D],
        "float32",
        "aligned native F32 state",
    )?;
    for input in [&qkv, &qk, &decay, &beta] {
        input.check(true)?;
    }
    state.check(false)?;
    out.check(false)?;

    let bad_qkv =
        Guarded::new(ctx, &bf_bytes(&fixture.qkv), &[rows, C], DType::BF16, 2)?;
    let bad_state =
        Guarded::new(ctx, &f32_bytes(&fixture.state), &[HV, D, D], DType::F32, 1)?;
    for (input, state_input) in
        [(&bad_qkv.view, &state.view), (&qkv.view, &bad_state.view)]
    {
        ensure!(
            !pipelined_prefill_buffers_aligned(input, &staging, state_input, &out.view),
            "misaligned fixture selected"
        );
        let pass = ctx.begin_concurrent()?;
        let result = gdn_prefill_pipelined_value_key_mlx(
            ctx,
            &pass,
            input,
            &staging,
            state_input,
            &out.view,
            GdnStateLayout::ValueKey,
        );
        ensure!(result.is_err(), "misaligned direct helper did not refuse");
        drop(pass);
    }
    bad_qkv.check(true)?;
    bad_state.check(true)?;
    described_is(
        out.view.raw_bytes(),
        &receipt["raw_y"],
        &[1, rows, HV, D],
        "bfloat16",
        "refusal preserved raw Y",
    )?;
    described_is(
        state.view.raw_bytes(),
        &receipt["state_out"],
        &[1, HV, D, D],
        "float32",
        "refusal preserved F32 state",
    )?;
    Ok(())
}

#[test]
fn official_pipelined_selector_preserves_reference_short_and_captured_paths() {
    for rows in [0, 1, 11, 12, 13, 63, 64, 65, 129, usize::MAX] {
        for layout in [GdnStateLayout::KeyValue, GdnStateLayout::ValueKey] {
            assert!(!value_key_pipelined_prefill_selected(false, layout, rows, false));
            assert!(!value_key_pipelined_prefill_selected(true, layout, rows, true));
            assert_eq!(
                value_key_pipelined_prefill_selected(true, layout, rows, false),
                layout == GdnStateLayout::ValueKey && rows >= 64
            );
        }
    }
}

#[test]
#[ignore = "requires the completed private stock oracle receipt and physical Metal"]
fn official_pipelined_direct_matches_stock_f32_state_and_bf16_readout() -> Result<()> {
    let receipt = oracle()?;
    let ctx = MetalContext::new()?;
    pipeline_metadata(&ctx)?;
    for (index, rows) in ROWS.into_iter().enumerate() {
        let fixture = direct_fixture(rows, index != 0);
        let expected = &receipt["fixtures"][index];
        let first = direct_once(&ctx, &fixture, expected)?;
        let repeat = direct_once(&ctx, &fixture, expected)?;
        ensure!(first == repeat, "pipelined native repeat hash mismatch");
        if index == 0 {
            aligned_views_and_refusals(&ctx, &fixture, expected)?;
        }
    }
    Ok(())
}

struct SparseFixture {
    qkv: Vec<f32>,
    history: Vec<f32>,
    weights: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    state: Vec<f32>,
}

impl SparseFixture {
    fn check_inputs(&self, receipt: &Value) -> Result<()> {
        for (name, values) in [
            ("qkv_bf16_sha256", &self.qkv),
            ("history_bf16_sha256", &self.history),
            ("weights_bf16_sha256", &self.weights),
            ("a_bf16_sha256", &self.a),
            ("b_bf16_sha256", &self.b),
            ("dt_bf16_sha256", &self.dt),
        ] {
            input_is(receipt, name, &bf_bytes(values))?;
        }
        for (name, values) in
            [("a_log_f32_sha256", &self.a_log), ("state_f32_sha256", &self.state)]
        {
            input_is(receipt, name, &f32_bytes(values))?;
        }
        Ok(())
    }
}

fn sparse_fixture() -> SparseFixture {
    let mut qkv = vec![0.0; 65 * C];
    for row in 0..65 {
        for h in 0..2 * HK {
            qkv[row * C + h * D] = 1.0;
        }
        for h in 0..HV {
            qkv[row * C + 2 * HK * D + h * D] = [1.0, -1.0, 2.0, -2.0][(row + h) % 4];
        }
    }
    let mut state = vec![0.0; HV * D * D];
    for h in 0..HV {
        state[h * D * D] = (h as i32 % 13 - 6) as f32 / 997.0f32;
    }
    SparseFixture {
        qkv,
        history: vec![0.0; 3 * C],
        weights: (0..C).flat_map(|_| [0.0, 0.0, 0.0, 1.0]).collect(),
        a: vec![-80.0; 65 * HV],
        b: vec![0.0; 65 * HV],
        a_log: vec![0.0; HV],
        dt: vec![0.0; HV],
        state,
    }
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

fn separated_qk(interleaved: &Tensor) -> (Vec<u8>, Vec<u8>) {
    let bytes = interleaved.raw_bytes();
    let mut q = Vec::with_capacity(65 * HK * D * 2);
    let mut k = Vec::with_capacity(65 * HK * D * 2);
    for row in 0..65 {
        let offset = row * 2 * HK * D * 2;
        q.extend_from_slice(&bytes[offset..offset + HK * D * 2]);
        k.extend_from_slice(&bytes[offset + HK * D * 2..offset + 2 * HK * D * 2]);
    }
    (q, k)
}

fn sparse_once(ctx: &MetalContext, receipt: &Value) -> Result<Vec<String>> {
    const M: usize = 65;
    let fixture = sparse_fixture();
    fixture.check_inputs(receipt)?;
    let x = Tensor::from_f32_as_bf16(ctx, &fixture.qkv, &[M, C])?;
    let history_in = Tensor::from_f32_as_bf16(ctx, &fixture.history, &[C, 3])?;
    let history_out = Tensor::zeros(ctx, &[C, 3], DType::BF16)?;
    // Stock stores [channel,tap]; native conv consumes [tap,channel].
    let mut weights = Vec::with_capacity(4 * C);
    for tap in 0..4 {
        for ch in 0..C {
            weights.push(fixture.weights[ch * 4 + tap]);
        }
    }
    let weights = Tensor::from_f32_as_bf16(ctx, &weights, &[4, C])?;
    let conv = Tensor::zeros(ctx, &[M, C], DType::BF16)?;
    let a = Tensor::from_f32_as_bf16(ctx, &fixture.a, &[M, HV])?;
    let b = Tensor::from_f32_as_bf16(ctx, &fixture.b, &[M, HV])?;
    let a_log = Tensor::from_f32(ctx, &fixture.a_log, &[HV])?;
    let dt = Tensor::from_f32_as_bf16(ctx, &fixture.dt, &[HV])?;
    let state = Tensor::from_f32(ctx, &fixture.state, &[HV, D, D])?;
    let qk = Tensor::zeros(ctx, &[M, 2 * HK * D], DType::BF16)?;
    let g = Tensor::zeros(ctx, &[M, HV], DType::F32)?;
    let beta = Tensor::zeros(ctx, &[M, HV], DType::F32)?;
    let staging =
        GdnRegscanStaging { qk_norm: &qk, decay: &g, beta: &beta, chunk: None };
    let raw = Tensor::zeros(ctx, &[M, HV, D], DType::BF16)?;
    let z = Tensor::zeros(ctx, &[M, HV * D], DType::BF16)?;
    let norm_w = Tensor::from_f32(ctx, &vec![1.0; D], &[D])?;
    let norm = Tensor::zeros(ctx, &[M, HV * D], DType::BF16)?;
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
    ensure!(
        value_key_pipelined_prefill_selected(true, GdnStateLayout::ValueKey, M, false),
        "sparse production recurrence not selected"
    );
    ensure!(
        pipelined_prefill_buffers_aligned(&conv, &staging, &state, &raw),
        "sparse production buffers unaligned"
    );
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
    pass.level_barrier(&[&raw, &state])?;
    gated_rmsnorm_with_rounding(
        ctx,
        &pass,
        &raw,
        &z,
        &norm_w,
        &norm,
        1e-6f32,
        GdnGate::Sigmoid,
        true,
    )?;
    pass.commit()?.wait()?;
    let pre = &receipt["prework"];
    hash_is(
        qk.raw_bytes(),
        text(pre, "native_interleaved_qk_bf16_sha256")?,
        "sparse interleaved QK",
    )?;
    let (q, k) = separated_qk(&qk);
    described_is(&q, &pre["q"], &[1, M, HK, D], "bfloat16", "sparse Q")?;
    described_is(&k, &pre["k"], &[1, M, HK, D], "bfloat16", "sparse K")?;
    let mut v = Vec::with_capacity(M * HV * D * 2);
    for row in 0..M {
        let offset = row * C * 2;
        v.extend_from_slice(&conv.raw_bytes()[offset + 2 * HK * D * 2..offset + C * 2]);
    }
    described_is(&v, &pre["v"], &[1, M, HV, D], "bfloat16", "sparse convolved V")?;
    let history = stock_history(&history_out)?;
    described_is(
        &history,
        &pre["conv_out"],
        &[1, 3, C],
        "bfloat16",
        "sparse convolution history",
    )?;
    described_is(g.raw_bytes(), &pre["g"], &[1, M, HV], "float32", "sparse decay")?;
    hash_is(
        beta.raw_bytes(),
        text(pre, "native_beta_f32_sha256")?,
        "sparse widened beta",
    )?;
    described_is(
        &bf_bytes(&beta.to_f32()?),
        &pre["beta"],
        &[1, M, HV],
        "bfloat16",
        "sparse beta",
    )?;
    described_is(
        raw.raw_bytes(),
        &receipt["raw_y"],
        &[1, M, HV, D],
        "bfloat16",
        "sparse raw Y",
    )?;
    described_is(
        state.raw_bytes(),
        &receipt["state_out"],
        &[1, HV, D, D],
        "float32",
        "sparse F32 state",
    )?;
    described_is(
        norm.raw_bytes(),
        &receipt["norm_out"],
        &[1, M, HV * D],
        "bfloat16",
        "sparse norm gate",
    )?;
    ensure!(
        state.to_f32()?.iter().all(|v| v.is_finite()),
        "sparse native state finite"
    );
    Ok([
        qk.raw_bytes(),
        &v,
        &history,
        g.raw_bytes(),
        beta.raw_bytes(),
        raw.raw_bytes(),
        state.raw_bytes(),
        norm.raw_bytes(),
    ]
    .iter()
    .map(|bytes| hex(&sha256(bytes)))
    .collect())
}

#[test]
#[ignore = "requires the completed private stock oracle receipt and physical Metal"]
fn official_pipelined_sparse_chain_matches_bounded_stock_prework_and_norm() -> Result<()>
{
    // This intentionally sparse TAB1 fixture qualifies the staged plumbing;
    // it does not qualify dense prework or dense norm reduction arithmetic.
    let receipt = oracle()?;
    let ctx = MetalContext::new()?;
    pipeline_metadata(&ctx)?;
    let first = sparse_once(&ctx, &receipt["fixtures"][7])?;
    let repeat = sparse_once(&ctx, &receipt["fixtures"][7])?;
    ensure!(first == repeat, "pipelined sparse native repeat hash mismatch");
    Ok(())
}
