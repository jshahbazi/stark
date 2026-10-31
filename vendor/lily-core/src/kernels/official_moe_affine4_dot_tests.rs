// SPDX-License-Identifier: Apache-2.0
// Test-only compact E10 arithmetic controls, not E512 host admission proof.
// Raw arrays stay in RAM; output contains only hashes/counts/aggregate errors.
use super::*;
use crate::metal::MslMathMode;
use crate::sha256::{hex, sha256};

const H: usize = 2560;
const I: usize = 640;
const E: usize = 10;
const TOPK: usize = 10;
const SEED: usize = 0x384A4D;
const IDS: [u32; 10] = [3, 8, 0, 7, 9, 1, 6, 5, 4, 2];
const REPORT_SHA: &str =
    "ce0f7dc7595646aee6a53fb450ecd03dd315ef5043fa932cdc4b836e96dcea62";
const CONTROLLER_SHA: &str =
    "91fea524e26d28eadbbd8d25e14bcf19c20a75522594ff2b8bead91d0a9d9d45";
const SHADER_SHA: &str =
    "5820e4a8467f329fea972e8f23b13d086faacf939d3e3005beaf02dc1c4a4971";
const TABLE_SHA: &str =
    "c487ccb3208e0ad603a280dd5b17a28910f9487929a0ba9bdfb1f782b3722b32";
const STOCK: [&str; 3] = [
    "489168f7ab401bf10eb4a52b585abac5f211317aa440031b395644b4cdbae91e",
    "fd837421f38ec82bd7fce93c3c43f3f55f6a6fa16432753cf05cf4dd705e367c",
    "8f13ca592f5903b4a5628ca9742db2c86fe3f2204cdb95966f74788dcfa06d99",
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

fn receipt() -> Result<serde_json::Value> {
    let path = std::env::var_os("STARK_QWEN38_MOE_DOT_ORACLE")
        .ok_or_else(|| anyhow::anyhow!("set private affine4 MoE dot receipt"))?;
    let supplied = std::env::var("STARK_QWEN38_MOE_DOT_ORACLE_SHA256")
        .map_err(|_| anyhow::anyhow!("set private affine4 MoE dot receipt SHA256"))?;
    ensure!(supplied == REPORT_SHA, "affine4 MoE report SHA256 guard");
    let path = std::path::Path::new(&path);
    let meta = std::fs::metadata(path)
        .map_err(|_| anyhow::anyhow!("affine4 MoE receipt unavailable"))?;
    ensure!(meta.is_file() && meta.len() <= 1024 * 1024, "affine4 MoE receipt size");
    let raw = std::fs::read(path)
        .map_err(|_| anyhow::anyhow!("affine4 MoE receipt unreadable"))?;
    ensure!(digest(&raw) == REPORT_SHA, "affine4 MoE report identity");
    let r: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|_| anyhow::anyhow!("affine4 MoE receipt format"))?;
    ensure!(
        r["schema"] == "stark-private-stock-affine4-MoE-dot-oracle-v1"
            && r["status"] == "complete"
            && r["controller_sha256"] == CONTROLLER_SHA
            && r["device"] == "gpu"
            && r["model_loaded"] == false
            && r["quantization_performed"] == false
            && r["raw_arrays_persisted"] == false
            && r["actual_Zig_dispatch_executed"] == false
            && r["full_E512_admission_or_addressing_qualified"] == false
            && r["forced_down_inputs_independent_of_gateup"] == true
            && r["all_repeats_exact"] == true
            && r["fixture_seed"] == SEED,
        "unqualified affine4 MoE reference"
    );
    ensure!(
        r["geometry"]
            == serde_json::json!({"H":H,"I":I,"ROWS":1,"SGS":2,"bits":4,"compact_experts":E,"group_size":64,"topK":TOPK})
            && r["affine4_zig_sha256"]
                == "71c2eec9275c761968703d94d19e750e4595c43462561149742da40b6aa047de"
            && r["stock_asset_sha256"]["lib/libmlx.dylib"]
                == "fc504cec0f86f4a20b111d60dd1112df178bfd7d9b7978d4d735fa1a9c9bf8d4"
            && r["stock_asset_sha256"]["lib/libmlxc.dylib"]
                == "83ebedc48c226376af608b939e17256550d4baa1ad0cc276195c676fdc55c501"
            && r["stock_asset_sha256"]["lib/mlx.metallib"]
                == "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1"
            && r["sigmoid_finite_sha256"] == TABLE_SHA,
        "affine4 MoE reference geometry/source/assets"
    );
    ensure!(
        digest(include_bytes!("metal/official_moe.metal")) == SHADER_SHA,
        "unchanged production MoE shader required"
    );
    let stages =
        r["stages"].as_array().ok_or_else(|| anyhow::anyhow!("MoE stages missing"))?;
    ensure!(stages.len() == 3, "MoE stage count");
    for (j, name) in [
        "gate_up_swiglu",
        "forced_activation_down_shared_0",
        "forced_activation_down_shared_1",
    ]
    .into_iter()
    .enumerate()
    {
        ensure!(
            stages[j]["name"] == name
                && stages[j]["output"]["sha256"] == STOCK[j]
                && stages[j]["repeat_mismatch_count"] == 0
                && stages[j]["finite_count"] == if j == 0 { TOPK * I } else { H },
            "MoE exact stock stage binding"
        );
    }
    Ok(r)
}

fn guarded_bf(ctx: &MetalContext, values: &[f32], shape: &[usize]) -> Result<Tensor> {
    let mut all = vec![17.0f32; 4];
    all.extend(values);
    all.extend([17.0; 4]);
    Tensor::from_f32_as_bf16(ctx, &all, &[all.len()])?.view(4, shape)
}
fn bank(ctx: &MetalContext, kind: usize, n: usize, k: usize) -> Result<QuantWeights> {
    let mut codes = vec![17u32; 4];
    for e in 0..E {
        for r in 0..n {
            for word in 0..k / 8 {
                let mut packed = 0u32;
                for j in 0..8 {
                    let c = word * 8 + j;
                    packed |=
                        (((e * 11 + r * 7 + c * 3 + (c / 64) * 5 + kind * 17 + SEED)
                            % 16) as u32)
                            << (4 * j);
                }
                codes.push(packed);
            }
        }
    }
    codes.extend([17; 4]);
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for e in 0..E {
        for r in 0..n {
            for g in 0..k / 64 {
                scales.push(bf(((e * 13 + r * 11 + g * 5 + kind * 3 + SEED) % 31 + 1)
                    as f32
                    / 32768.0));
                biases.push(bf((((e * 3 + r * 7 + g * 11 + kind * 5 + SEED) % 29)
                    as i32
                    - 14) as f32
                    / 8192.0));
            }
        }
    }
    Ok(QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[codes.len()],
            DType::U32,
        )?
        .view(4, &[E * n, k / 8])?,
        scales: guarded_bf(ctx, &scales, &[E * n, k / 64])?,
        biases: guarded_bf(ctx, &biases, &[E * n, k / 64])?,
        bits: 4,
        group_size: 64,
    })
}
fn output(ctx: &MetalContext, n: usize) -> Result<(Tensor, Tensor)> {
    let storage = Tensor::from_f32_as_bf16(ctx, &vec![17.0; n + 8], &[n + 8])?;
    let out = storage.view(4, &[n])?;
    Ok((storage, out))
}
fn ordered(v: u16) -> u32 {
    if v & 0x8000 != 0 { u32::from(!v) } else { u32::from(v | 0x8000) }
}

#[test]
#[ignore = "requires physical Metal and the exact private stock affine4 dot receipt"]
fn official_moe_affine4_collect_compiler_modes_and_require_current_stock_hashes()
-> Result<()> {
    // Receipt/source guards precede Metal creation; native input hashes are
    // checked before dispatch. Compact E10 launches unchanged production
    // entrypoints directly, rather than the E512 host wrappers.
    let oracle = receipt()?;
    let ctx = MetalContext::new()?;
    let gate = bank(&ctx, 0, I, H)?;
    let up = bank(&ctx, 1, I, H)?;
    let down = bank(&ctx, 2, H, I)?;
    for (j, w) in [&gate, &up, &down].into_iter().enumerate() {
        for (field, t) in
            [("codes", &w.codes), ("scales", &w.scales), ("biases", &w.biases)]
        {
            ensure!(
                oracle["banks"][j][field]["sha256"] == hash(t),
                "MoE bank input hash mismatch"
            );
        }
    }
    let xv: Vec<f32> = (0..H)
        .map(|i| bf((((i * 17 + SEED) % 101) as i32 - 50) as f32 / 71.0))
        .collect();
    let av: Vec<f32> = (0..TOPK * I)
        .map(|i| bf((((i * 29 + SEED) % 97) as i32 - 48) as f32 / 113.0))
        .collect();
    let sw: Vec<usize> = (0..TOPK).map(|i| (i * 7 + 3) % 19 + 1).collect();
    let sum: usize = sw.iter().sum();
    let sv: Vec<f32> = sw.iter().map(|v| bf(*v as f32 / sum as f32)).collect();
    let shv: Vec<f32> = (0..H)
        .map(|i| bf((((i * 13 + SEED) % 83) as i32 - 41) as f32 / 97.0))
        .collect();
    let x = guarded_bf(&ctx, &xv, &[H])?;
    let act = guarded_bf(&ctx, &av, &[TOPK, I])?;
    let score_bf = guarded_bf(&ctx, &sv, &[TOPK])?;
    let mut score_storage = vec![17.0f32; 4];
    score_storage.extend(&sv);
    score_storage.extend([17.0; 4]);
    let scores =
        Tensor::from_f32(&ctx, &score_storage, &[TOPK + 8])?.view(4, &[TOPK])?;
    let shared = guarded_bf(&ctx, &shv, &[H])?;
    let glog = guarded_bf(&ctx, &[-6.84375], &[1])?;
    for (field, t) in [
        ("x", &x),
        ("forced_act", &act),
        ("scores", &score_bf),
        ("shared_down", &shared),
        ("shared_gate", &glog),
    ] {
        ensure!(
            oracle["operands"][field]["sha256"] == hash(t),
            "MoE operand input hash mismatch"
        );
    }
    ensure!(
        scores.to_f32()? == score_bf.to_f32()?,
        "BF16 score widening must be lossless"
    );
    let mut iv = vec![17u32; 4];
    iv.extend(IDS);
    iv.extend([17; 4]);
    let indices =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&iv), &[TOPK + 8], DType::U32)?
            .view(4, &[TOPK])?;
    ensure!(
        oracle["indices"]["sha256"] == hash(&indices),
        "MoE expert-order input hash mismatch"
    );
    let table = official_moe_sigmoid_table(&ctx)?;
    let finite: Vec<u8> = (0..=u16::MAX)
        .filter(|b| b & 0x7f80 != 0x7f80)
        .flat_map(|b| {
            table.raw_bytes()[b as usize * 2..b as usize * 2 + 2].iter().copied()
        })
        .collect();
    ensure!(digest(&finite) == TABLE_SHA, "native production sigmoid table identity");
    let mut current_values: Vec<Vec<u16>> = Vec::new();
    let mut matched_modes = [0usize; 3];
    let mut all_repeats = true;
    let mut all_guards = true;
    let mut all_finite = true;
    for (mode, (label, math, version)) in [
        ("current_Default3_1", MslMathMode::PlatformDefault, MslVersion::V3_1),
        ("diagnostic_Safe3_1", MslMathMode::Safe, MslVersion::V3_1),
        ("diagnostic_Safe4_0", MslMathMode::Safe, MslVersion::V4_0),
    ]
    .into_iter()
    .enumerate()
    {
        let gp = ctx.pipeline_with_math_mode(
            "official_moe_gate_up_a4",
            OFFICIAL_MOE_SOURCE,
            version,
            math,
        )?;
        let dp = ctx.pipeline_with_math_mode(
            "official_moe_down_a4",
            OFFICIAL_MOE_SOURCE,
            version,
            math,
        )?;
        ensure!(
            gp.thread_execution_width() == 32 && dp.thread_execution_width() == 32,
            "MoE stock SIMD width32 required"
        );
        let outputs = [output(&ctx, TOPK * I)?, output(&ctx, H)?, output(&ctx, H)?];
        let repeats = [output(&ctx, TOPK * I)?, output(&ctx, H)?, output(&ctx, H)?];
        let pass = ctx.begin_concurrent()?;
        for out in [&outputs[0].1, &repeats[0].1] {
            pass.dispatch_at(
                &gp,
                &[
                    gate.codes.binding(),
                    gate.scales.binding(),
                    gate.biases.binding(),
                    up.codes.binding(),
                    up.scales.binding(),
                    up.biases.binding(),
                    x.binding(),
                    indices.binding(),
                    table.binding(),
                    out.binding(),
                ],
                &[&u32_bytes(H), &u32_bytes(I), &u32_bytes(64)],
                Grid::Threadgroups {
                    groups: (I / 2, TOPK, 1),
                    threadgroup: (64, 1, 1),
                },
            )?;
        }
        for has_shared in 0..=1 {
            for out in [&outputs[has_shared + 1].1, &repeats[has_shared + 1].1] {
                pass.dispatch_at(
                    &dp,
                    &[
                        down.codes.binding(),
                        down.scales.binding(),
                        down.biases.binding(),
                        act.binding(),
                        indices.binding(),
                        scores.binding(),
                        shared.binding(),
                        glog.binding(),
                        out.binding(),
                    ],
                    &[
                        &u32_bytes(I),
                        &u32_bytes(H),
                        &u32_bytes(64),
                        &u32_bytes(TOPK),
                        &u32_bytes(has_shared),
                    ],
                    Grid::Threadgroups {
                        groups: (H / 2, 1, 1),
                        threadgroup: (64, 1, 1),
                    },
                )?;
            }
        }
        pass.commit()?.wait()?;
        for stage in 0..3 {
            let out = &outputs[stage].1;
            let actual = hash(out);
            let n = out.numel();
            let repeated = actual == hash(&repeats[stage].1);
            all_repeats &= repeated;
            let mut guards = true;
            for pair in [&outputs[stage], &repeats[stage]] {
                let all = pair.0.to_f32()?;
                guards &= all[..4].iter().chain(&all[n + 4..]).all(|v| *v == 17.0);
            }
            all_guards &= guards;
            let values: Vec<u16> =
                bytemuck::cast_slice::<u8, u16>(out.raw_bytes()).to_vec();
            let finite = values.iter().filter(|b| **b & 0x7f80 != 0x7f80).count();
            all_finite &= finite == n;
            let stock_match = actual == STOCK[stage];
            matched_modes[mode] += usize::from(stock_match);
            if mode == 0 {
                current_values.push(values.clone());
            }
            let changed = values
                .iter()
                .zip(&current_values[stage])
                .filter(|(a, b)| a != b)
                .count();
            let max_ulp = values
                .iter()
                .zip(&current_values[stage])
                .map(|(a, b)| ordered(*a).abs_diff(ordered(*b)))
                .max()
                .unwrap_or(0);
            println!(
                "moe_affine4_dot mode={label} stage={stage} exact_stock_hash_match={stock_match} repeat_exact={repeated} guards_exact={guards} finite_count={finite} changed_vs_current_count={changed} max_BF16_ulps_vs_current={max_ulp} output_BF16_sha256={actual}"
            );
        }
    }
    println!(
        "moe_affine4_dot_summary current_matches={}/3 diagnostic_Safe3_1_matches={}/3 diagnostic_Safe4_0_matches={}/3 compact_experts=10 full_E512_admission_qualified=false",
        matched_modes[0], matched_modes[1], matched_modes[2]
    );
    // No numerical early return: every mode/stage is collected before failing.
    ensure!(
        all_repeats && all_guards && all_finite,
        "MoE repeat/canary/finite control failure after all modes collected"
    );
    ensure!(
        matched_modes[0] == 3,
        "current production Default3.1 strict stock gate failed after all modes: matched_stages={}/3; alternative modes do not replace current acceptance",
        matched_modes[0]
    );
    Ok(())
}
