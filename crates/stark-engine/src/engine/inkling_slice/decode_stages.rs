// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only decode stage table: GPU milliseconds, launches and weight
//! bytes per stage for one step and for two- to four-row verifier rounds,
//! with measured read bandwidth ceilings. Never prints tokens or values.

use std::{collections::BTreeMap, path::Path};

use serde_json::{Value, json};

use super::{
    affine8_validation::{ingest, load_worker_session, median},
    *,
};
use crate::engine::inkling_slice::affine8;

const STAGES: [&str; 9] =
    ["dense", "attention", "experts", "attention_inputs", "expert_inputs", "expert_finish", "readout", "other", "gap"];
const TOP: usize = 8;
const MODES: [&str; 2] = ["read", "read_sum"];
const END_OF_SAMPLING: u32 = 200_006;
const CONTENT_TEXT: u32 = 200_004;
const DECODE_UP: Kernel = "ink_nvfp4_decode_up";
pub(in crate::engine::inkling_slice) const DECODE_EXTRA: [Kernel; 3] =
    ["ink_dx_up_hb", "ink_dx_down_hb", "ink_dx_to_half"];
/// Multi-channel single-row NVFP4 kernels: (up, down, channels per SIMD group).
pub(in crate::engine::inkling_slice) const DECODE_VARIANTS: [(Kernel, Kernel, usize); 6] = [
    (DECODE_UP, "ink_nvfp4_decode_down", 4),
    ("ink_dx_up_c1", "ink_dx_down_c1", 1),
    ("ink_dx_up_c2", "ink_dx_down_c2", 2),
    ("ink_dx_up_c1h", "ink_dx_down_c1h", 1),
    ("ink_dx_up_c2h", "ink_dx_down_c2h", 2),
    ("ink_dx_up_c4h", "ink_dx_down_c4h", 4),
];

fn stage_of(phase: PrefillPhase) -> &'static str {
    match phase {
        PrefillPhase::AttentionDense => "dense",
        PrefillPhase::RelativeCacheAttention
        | PrefillPhase::KeyValue
        | PrefillPhase::RelativeTable
        | PrefillPhase::LocalAttention
        | PrefillPhase::GlobalAttention
        | PrefillPhase::GlobalPacking
        | PrefillPhase::GlobalMask
        | PrefillPhase::GlobalGate => "attention",
        PrefillPhase::AttentionInputs => "attention_inputs",
        PrefillPhase::ExpertInputs => "expert_inputs",
        PrefillPhase::ExpertFinish => "expert_finish",
        PrefillPhase::Readout => "readout",
        PrefillPhase::Other => "other",
        _ => "experts",
    }
}

fn stage_ms(report: &Value) -> BTreeMap<&'static str, f64> {
    let get = |key: &str| report[key].as_f64().unwrap_or(0.0);
    BTreeMap::from([
        ("dense", get("attention_dense_ms")),
        ("attention", get("relative_cache_attention_ms")),
        ("experts", get("experts_ms")),
        ("attention_inputs", get("attention_inputs_ms")),
        ("expert_inputs", get("expert_inputs_ms")),
        ("expert_finish", get("expert_finish_ms")),
        ("readout", get("readout_ms")),
        ("other", get("embedding_readout_ms") - get("readout_ms")),
        ("gap", get("command_gap_ms")),
    ])
}

/// Bytes of one expert across its up and down planes, and the planes.
fn expert_planes(layer: &Layer) -> Vec<&Buffer> {
    match &layer.experts {
        Experts::Nvfp4 {
            up,
            down,
        } => vec![&up.codes, &up.blocks, &down.codes, &down.blocks],
        Experts::Mxfp8 {
            up,
            down,
        } => vec![&up.codes, &up.exponents, &down.codes, &down.exponents],
    }
}

fn per_expert(buffer: &Buffer) -> usize {
    buffer.length() / EXPERTS
}

/// Dense projection weight bytes of one visit, as stored for decode.
fn dense_bytes(
    session: &Session,
    layer: &Layer,
) -> usize {
    let mut tensors = vec![&layer.q, &layer.k, &layer.v, &layer.r, &layer.attn_gate, &layer.sink, &layer.output];
    if let Some(dynamic) = &layer.dynamic {
        tensors.extend(dynamic.iter());
    }
    tensors
        .into_iter()
        .map(|copy| {
            session
                .runtime
                .affine8
                .as_ref()
                .and_then(|weights| weights.tensor_for(copy))
                .map_or(copy.length(), affine8::Affine8::bytes)
        })
        .sum()
}

/// Per-stage GPU time, launches and bytes on the restored prompt state.
pub fn measure_decode_stages(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    repeats: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = load_worker_session(checkpoint, Some(draft), context)?;
    affine8::compile(
        &mut session.runtime,
        include_str!("decode_probe.metal"),
        &["ink_probe_read", "ink_probe_gather"],
    )?;
    ingest(&mut session, &tokens, prompt)?;
    let anchor = transaction_validation::greedy(&session.step(tokens[prompt - 1])?);
    let base = session.snapshot()?;
    // Serial continuation from the base: the verifier rows and their routes.
    let mut inputs = vec![anchor];
    let mut routes = Vec::new();
    for step in 0..MAX_VERIFY_ROWS {
        let logits = session.step(inputs[step])?;
        let ids = session.visits.iter().map(|visit| read_shared::<u32>(&visit.scratch.routes, TOP)).collect::<Vec<_>>();
        routes.push(ids);
        if step + 1 < MAX_VERIFY_ROWS {
            inputs.push(transaction_validation::greedy(&logits));
        }
    }
    let routes_valid = routes.iter().flatten().flatten().all(|&id| (id as usize) < EXPERTS);
    let distinct = |rows: usize, visit: usize| {
        let mut ids = (0..rows).flat_map(|row| routes[row][visit].iter().copied()).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        ids
    };

    let mut profile = PrefillTiming::new(&session.runtime)?;
    profile.fold_verify_phases = true;
    let mut table = Vec::new();
    for rows in 1..=MAX_VERIFY_ROWS {
        let mut ms: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut launches: BTreeMap<&str, usize> = BTreeMap::new();
        let mut plain = Vec::new();
        let mut profiled_total = Vec::new();
        let mut dispatches = 0;
        for repeat in 0..repeats.max(1) {
            for instrumented in [repeat % 2 == 0, repeat % 2 == 1] {
                session.restore(&base)?;
                let timing = if rows == 1 {
                    let result = session.encode_append(&[anchor], true, instrumented.then_some(&mut profile));
                    let summary = result.as_ref().map(|t| (t.gpu_ms, t.dispatches)).map_err(|e| *e);
                    session.finish_append(result, 1)?;
                    summary?
                } else {
                    let (txn, timing) =
                        session.begin_verify_profiled(&inputs[..rows], instrumented.then_some(&mut profile))?;
                    drop(txn);
                    (timing.gpu_ms, timing.dispatches)
                };
                if !instrumented {
                    plain.push(timing.0);
                    dispatches = timing.1;
                    continue;
                }
                profiled_total.push(timing.0);
                for (stage, value) in stage_ms(&profile.command_report(timing.0)) {
                    ms.entry(stage).or_default().push(value);
                }
                launches.clear();
                for &(phase, count) in &profile.span_dispatches {
                    *launches.entry(stage_of(phase)).or_default() += count;
                }
            }
        }
        // Bytes per stage.
        let mut bytes: BTreeMap<&str, f64> = BTreeMap::new();
        let position = prompt;
        for (index, (&physical, visit)) in PHYSICAL.iter().zip(&session.visits).enumerate() {
            let layer = &session.layers[physical];
            *bytes.entry("dense").or_default() += dense_bytes(&session, layer) as f64;
            let experts = if routes_valid {
                distinct(rows, index).len()
            } else {
                (rows * TOP).min(EXPERTS)
            };
            let expert = expert_planes(layer).into_iter().map(per_expert).sum::<usize>();
            *bytes.entry("experts").or_default() += (experts * expert + layer.router.length()) as f64;
            let per_position = (visit.keys.length() + visit.values.length()) as f64 / visit.capacity as f64;
            *bytes.entry("attention").or_default() += per_position * (position + rows).min(visit.capacity) as f64;
            if (18..42).contains(&index) {
                let gate = |slot: usize| {
                    let gate = &layer.gates[slot];
                    (gate.down.length() + gate.gain.length() + gate.forget.length()) as f64
                };
                *bytes.entry("attention_inputs").or_default() += gate(0) + gate(1) + gate(2);
                *bytes.entry("expert_inputs").or_default() += gate(3);
            }
        }
        let head =
            session.runtime.affine8.as_ref().map_or(session.unembedding.length(), |weights| weights.head.bytes());
        *bytes.entry("readout").or_default() += head as f64;
        let mut stages = serde_json::Map::new();
        for stage in STAGES {
            let mut values = ms.get(stage).cloned().unwrap_or_default();
            let gpu = median(&mut values);
            let moved = bytes.get(stage).copied().unwrap_or(0.0);
            stages.insert(
                stage.into(),
                json!({
                    "gpu_ms": gpu,
                    "launches": launches.get(stage).copied().unwrap_or(0),
                    "weight_bytes": moved,
                    "effective_gb_s": if gpu > 0.0 && moved > 0.0 { moved / gpu / 1e6 } else { 0.0 },
                }),
            );
        }
        table.push(json!({
            "rows": rows,
            "uninstrumented_gpu_ms": median(&mut plain),
            "instrumented_gpu_ms": median(&mut profiled_total),
            "launches": dispatches,
            "distinct_experts_per_visit_mean": if routes_valid {
                (0..PHYSICAL.len()).map(|v| distinct(rows, v).len()).sum::<usize>() as f64 / PHYSICAL.len() as f64
            } else {
                f64::NAN
            },
            "stages": stages,
        }));
    }

    // Expert sub-stages: one row with unfolded phases, four rows with the
    // verifier's expert phases split out.
    let mut detail = Vec::new();
    for rows in [1, MAX_VERIFY_ROWS] {
        let mut fine = PrefillTiming::new(&session.runtime)?;
        fine.split_verify_experts = rows > 1;
        for _ in 0..3 {
            session.restore(&base)?;
            let gpu_ms = if rows == 1 {
                let result = session.encode_append(&[anchor], true, Some(&mut fine));
                let summary = result.as_ref().map(|t| t.gpu_ms).map_err(|e| *e);
                session.finish_append(result, 1)?;
                summary?
            } else {
                let (txn, timing) = session.begin_verify_profiled(&inputs[..rows], Some(&mut fine))?;
                drop(txn);
                timing.gpu_ms
            };
            let mut launches: BTreeMap<String, usize> = BTreeMap::new();
            for &(phase, count) in &fine.span_dispatches {
                *launches.entry(format!("{}", phase as usize)).or_default() += count;
            }
            detail.push(
                json!({"rows": rows, "report": fine.command_report(gpu_ms), "launches_by_phase_index": launches}),
            );
        }
    }

    // Ceilings: streaming reads of large buffers and gathered reads of the
    // experts each row set selects, read-only and read-and-sum.
    let out = zeroed(&session.runtime.device, 64)?;
    let mut ceilings = Vec::new();
    let streams: Vec<(&str, &Buffer)> = match session.runtime.affine8.as_ref() {
        Some(weights) => vec![("affine8_head", &weights.head.tensor), ("bf16_head_copy", &session.unembedding)],
        None => vec![("bf16_head", &session.unembedding)],
    };
    for (name, buffer) in streams {
        for mode in [0u32, 1] {
            let mut values = Vec::new();
            for _ in 0..repeats.max(3) {
                let count = buffer.length() / 16;
                values.push(
                    execute(&session.runtime, |gpu| {
                        gpu.dispatch(
                            "ink_probe_read",
                            &[binding(buffer), binding(&out)],
                            [count as u32, mode],
                            (4096, 1),
                            256,
                        );
                    })?
                    .gpu_ms,
                );
            }
            let gpu = median(&mut values);
            ceilings.push(json!({"buffer": name, "mode": MODES[mode as usize], "bytes": buffer.length(),
                "gpu_ms": gpu, "gb_s": buffer.length() as f64 / gpu / 1e6}));
        }
    }
    if routes_valid {
        for rows in [1, MAX_VERIFY_ROWS] {
            let lists = (0..PHYSICAL.len()).map(|visit| distinct(rows, visit)).collect::<Vec<_>>();
            let flat = lists.iter().flatten().copied().collect::<Vec<u32>>();
            let ids = zeroed(&session.runtime.device, flat.len() * 4)?;
            unsafe { std::slice::from_raw_parts_mut(ids.contents().as_ptr().cast::<u32>(), flat.len()) }
                .copy_from_slice(&flat);
            let mut moved = 0usize;
            for (list, &physical) in lists.iter().zip(&PHYSICAL) {
                moved +=
                    list.len() * expert_planes(&session.layers[physical]).into_iter().map(per_expert).sum::<usize>();
            }
            for mode in [0u32, 1] {
                let mut values = Vec::new();
                for _ in 0..repeats.max(3) {
                    values.push(
                        execute(&session.runtime, |gpu| {
                            let mut first = 0u32;
                            for (list, &physical) in lists.iter().zip(&PHYSICAL) {
                                for plane in expert_planes(&session.layers[physical]) {
                                    let per = per_expert(plane) / 16;
                                    gpu.dispatch(
                                        "ink_probe_gather",
                                        &[binding(plane), binding(&ids), binding(&out)],
                                        [per as u32, list.len() as u32, mode, first],
                                        (per.div_ceil(256 * 8).max(1), list.len()),
                                        256,
                                    );
                                }
                                first += list.len() as u32;
                            }
                        })?
                        .gpu_ms,
                    );
                }
                let gpu = median(&mut values);
                ceilings.push(json!({"buffer": format!("selected_experts_{rows}_rows"), "mode": MODES[mode as usize],
                    "bytes": moved, "launches": PHYSICAL.len() * 4, "gpu_ms": gpu, "gb_s": moved as f64 / gpu / 1e6}));
            }
        }
    }
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "affine8": session.runtime.affine8.is_some(),
        "routes_read_from_decode": routes_valid,
        "note": "instrumented stage times split encoders at stage boundaries; uninstrumented is the whole command",
        "table": table,
        "expert_detail": detail,
        "ceilings": ceilings,
    }))
}

/// Encode the single-row NVFP4 up and down products of every NVFP4 visit with
/// the incumbent kernels (`variant` None) or a multi-channel variant.
fn encode_nvfp4_products(
    gpu: &mut Stream<'_>,
    session: &Session,
    variant: Option<(Kernel, Kernel, usize)>,
    stage: Option<bool>,
) {
    for (&physical, visit) in PHYSICAL.iter().zip(&session.visits) {
        let Experts::Nvfp4 {
            up,
            down,
        } = &session.layers[physical].experts
        else {
            continue;
        };
        let s = &visit.scratch;
        if stage != Some(false) {
            let (kernel, groups) = variant.map_or(("ink_nvfp4_vector_up", MID / 4), |v| (v.0, MID / (4 * v.2)));
            gpu.one(
                kernel,
                &[&s.up_inner, &s.up_max, &s.routes, &s.route_weights, &up.codes, &up.blocks, &up.maxima, &s.mid],
                up.args(),
                (groups, 8),
                128,
            );
        }
        if stage != Some(true) {
            let (kernel, groups) = variant.map_or(("ink_nvfp4_vector_down", WIDTH / 4), |v| (v.1, WIDTH / (4 * v.2)));
            gpu.one(
                kernel,
                &[&s.down_inner, &s.down_max, &s.routes, &down.codes, &down.blocks, &down.maxima, &s.down],
                down.args(),
                (groups, 8),
                128,
            );
        }
    }
}

/// Multi-channel single-row NVFP4 kernel variants against the incumbent ones
/// on restored real state: bitwise equality of every output and paired timing.
pub fn measure_decode_experts(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    pairs: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = load_worker_session(checkpoint, Some(draft), context)?;
    if !session.runtime.nvfp4_vector || !session.runtime.pipelines.contains_key(DECODE_UP) {
        return Err("Inkling decode expert kernels are not registered");
    }
    ingest(&mut session, &tokens, prompt)?;
    session.step(tokens[prompt - 1])?;
    // The direct default path skips the activation buffers; fill them from
    // the step's expert inputs so the incumbent kernels read real state.
    execute(&session.runtime, |gpu| {
        for (&physical, visit) in PHYSICAL.iter().zip(&session.visits) {
            if matches!(session.layers[physical].experts, Experts::Nvfp4 { .. }) {
                let s = &visit.scratch;
                gpu.one("ink_nvfp4_activation", &[&s.mlp_input, &s.up_inner, &s.up_max], WIDTH as u32, (1, 1), 256);
                gpu.one("ink_nvfp4_activation", &[&s.mid, &s.down_inner, &s.down_max], MID as u32, (8, 1), 256);
            }
        }
    })?;
    let read = |session: &Session, stage: bool| {
        session
            .visits
            .iter()
            .zip(&PHYSICAL)
            .filter(|(_, physical)| matches!(session.layers[**physical].experts, Experts::Nvfp4 { .. }))
            .map(|(visit, _)| {
                if stage {
                    read_shared::<u16>(&visit.scratch.mid, 8 * MID)
                } else {
                    read_shared::<u16>(&visit.scratch.down, 8 * WIDTH)
                }
            })
            .collect::<Vec<_>>()
    };
    let mut reports = Vec::new();
    for variant in DECODE_VARIANTS {
        let mut unequal = [0usize; 2];
        let mut values = [0usize; 2];
        for stage in [true, false] {
            execute(&session.runtime, |gpu| encode_nvfp4_products(gpu, &session, None, Some(stage)))?;
            let want = read(&session, stage);
            execute(&session.runtime, |gpu| encode_nvfp4_products(gpu, &session, Some(variant), Some(stage)))?;
            let got = read(&session, stage);
            let slot = usize::from(!stage);
            for (a, b) in want.iter().zip(&got) {
                values[slot] += a.len();
                unequal[slot] += a.iter().zip(b).filter(|(x, y)| x != y).count();
            }
        }
        let mut timing = Vec::new();
        for pair in 0..pairs {
            let order = if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let mut ms = [0.0; 2];
            for candidate in order {
                let chosen = candidate.then_some(variant);
                ms[usize::from(candidate)] =
                    execute(&session.runtime, |gpu| encode_nvfp4_products(gpu, &session, chosen, None))?.gpu_ms;
            }
            timing.push((ms[0], ms[1]));
        }
        reports.push(json!({
            "up": variant.0, "down": variant.1, "channels": variant.2,
            "up_values": values[0], "up_unequal": unequal[0],
            "down_values": values[1], "down_unequal": unequal[1],
            "incumbent_vs_variant_ms": super::affine8_validation::timing_summary(&timing),
        }));
    }
    // Half copies of the activations for the half-buffer kernels.
    let device = session.runtime.device.clone();
    let halves = session
        .visits
        .iter()
        .map(|_| Ok((zeroed(&device, WIDTH * 2)?, zeroed(&device, 8 * MID * 2)?)))
        .collect::<Result<Vec<_>>>()?;
    execute(&session.runtime, |gpu| {
        for (visit, (up_half, down_half)) in session.visits.iter().zip(&halves) {
            let s = &visit.scratch;
            gpu.one("ink_dx_to_half", &[&s.up_inner, up_half], WIDTH as u32, (WIDTH.div_ceil(128), 1), 128);
            gpu.one("ink_dx_to_half", &[&s.down_inner, down_half], (8 * MID) as u32, ((8 * MID).div_ceil(128), 1), 128);
        }
    })?;
    let encode_half = |gpu: &mut Stream<'_>, session: &Session, stage: Option<bool>| {
        for ((&physical, visit), (up_half, down_half)) in PHYSICAL.iter().zip(&session.visits).zip(&halves) {
            let Experts::Nvfp4 {
                up,
                down,
            } = &session.layers[physical].experts
            else {
                continue;
            };
            let s = &visit.scratch;
            if stage != Some(false) {
                gpu.one(
                    "ink_dx_up_hb",
                    &[up_half, &s.up_max, &s.routes, &s.route_weights, &up.codes, &up.blocks, &up.maxima, &s.mid],
                    up.args(),
                    (MID / 4, 8),
                    128,
                );
            }
            if stage != Some(true) {
                gpu.one(
                    "ink_dx_down_hb",
                    &[down_half, &s.down_max, &s.routes, &down.codes, &down.blocks, &down.maxima, &s.down],
                    down.args(),
                    (WIDTH / 4, 8),
                    128,
                );
            }
        }
    };
    let mut half_unequal = [0usize; 2];
    for stage in [true, false] {
        execute(&session.runtime, |gpu| encode_nvfp4_products(gpu, &session, None, Some(stage)))?;
        let want = read(&session, stage);
        execute(&session.runtime, |gpu| encode_half(gpu, &session, Some(stage)))?;
        let got = read(&session, stage);
        for (a, b) in want.iter().zip(&got) {
            half_unequal[usize::from(!stage)] += a.iter().zip(b).filter(|(x, y)| x != y).count();
        }
    }
    let mut half_timing = Vec::new();
    for pair in 0..pairs {
        let order = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut ms = [0.0; 2];
        for candidate in order {
            ms[usize::from(candidate)] = if candidate {
                execute(&session.runtime, |gpu| encode_half(gpu, &session, None))?.gpu_ms
            } else {
                execute(&session.runtime, |gpu| encode_nvfp4_products(gpu, &session, None, None))?.gpu_ms
            };
        }
        half_timing.push((ms[0], ms[1]));
    }
    reports.push(json!({
        "up": "ink_dx_up_hb", "down": "ink_dx_down_hb", "channels": 1,
        "up_unequal": half_unequal[0], "down_unequal": half_unequal[1],
        "incumbent_vs_variant_ms": super::affine8_validation::timing_summary(&half_timing),
    }));
    // Direct BF16-input products against the activation passes plus the
    // incumbent products (both launches counted).
    if !QUANTIZE_EXPERT_INPUTS && super::DIRECT_PRODUCTS.iter().all(|name| session.runtime.pipelines.contains_key(name))
    {
        let encode_direct = |gpu: &mut Stream<'_>, session: &Session, stage: Option<bool>| {
            for (&physical, visit) in PHYSICAL.iter().zip(&session.visits) {
                let Experts::Nvfp4 {
                    up,
                    down,
                } = &session.layers[physical].experts
                else {
                    continue;
                };
                let s = &visit.scratch;
                if stage != Some(false) {
                    gpu.one(
                        super::DIRECT_PRODUCTS[0],
                        &[
                            &s.mlp_input,
                            &s.up_max,
                            &s.routes,
                            &s.route_weights,
                            &up.codes,
                            &up.blocks,
                            &up.maxima,
                            &s.mid,
                        ],
                        up.args(),
                        (MID / 4, 8),
                        128,
                    );
                }
                if stage != Some(true) {
                    gpu.one(
                        super::DIRECT_PRODUCTS[1],
                        &[&s.mid, &s.down_max, &s.routes, &down.codes, &down.blocks, &down.maxima, &s.down],
                        down.args(),
                        (WIDTH / 4, 8),
                        128,
                    );
                }
            }
        };
        let encode_incumbent = |gpu: &mut Stream<'_>, session: &Session, stage: Option<bool>| {
            for (&physical, visit) in PHYSICAL.iter().zip(&session.visits) {
                let Experts::Nvfp4 {
                    up,
                    down,
                } = &session.layers[physical].experts
                else {
                    continue;
                };
                let s = &visit.scratch;
                if stage != Some(false) {
                    gpu.one("ink_nvfp4_activation", &[&s.mlp_input, &s.up_inner, &s.up_max], WIDTH as u32, (1, 1), 256);
                    gpu.one(
                        "ink_nvfp4_vector_up",
                        &[
                            &s.up_inner,
                            &s.up_max,
                            &s.routes,
                            &s.route_weights,
                            &up.codes,
                            &up.blocks,
                            &up.maxima,
                            &s.mid,
                        ],
                        up.args(),
                        (MID / 4, 8),
                        128,
                    );
                }
                if stage != Some(true) {
                    gpu.one("ink_nvfp4_activation", &[&s.mid, &s.down_inner, &s.down_max], MID as u32, (8, 1), 256);
                    gpu.one(
                        "ink_nvfp4_vector_down",
                        &[&s.down_inner, &s.down_max, &s.routes, &down.codes, &down.blocks, &down.maxima, &s.down],
                        down.args(),
                        (WIDTH / 4, 8),
                        128,
                    );
                }
            }
        };
        let mut direct_unequal = [0usize; 2];
        for stage in [true, false] {
            execute(&session.runtime, |gpu| encode_incumbent(gpu, &session, Some(stage)))?;
            let want = read(&session, stage);
            execute(&session.runtime, |gpu| encode_direct(gpu, &session, Some(stage)))?;
            let got = read(&session, stage);
            for (a, b) in want.iter().zip(&got) {
                direct_unequal[usize::from(!stage)] += a.iter().zip(b).filter(|(x, y)| x != y).count();
            }
        }
        let mut direct_timing = Vec::new();
        for pair in 0..pairs {
            let order = if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let mut ms = [0.0; 2];
            for candidate in order {
                ms[usize::from(candidate)] = if candidate {
                    execute(&session.runtime, |gpu| encode_direct(gpu, &session, None))?.gpu_ms
                } else {
                    execute(&session.runtime, |gpu| encode_incumbent(gpu, &session, None))?.gpu_ms
                };
            }
            direct_timing.push((ms[0], ms[1]));
        }
        reports.push(json!({
            "up": super::DIRECT_PRODUCTS[0], "down": super::DIRECT_PRODUCTS[1], "channels": 1,
            "incumbent_includes_activation": true,
            "up_unequal": direct_unequal[0], "down_unequal": direct_unequal[1],
            "incumbent_vs_variant_ms": super::affine8_validation::timing_summary(&direct_timing),
        }));
    }
    Ok(json!({
        "prompt_tokens": prompt,
        "nvfp4_visits": PHYSICAL.iter().filter(|&&p| matches!(session.layers[p].experts, Experts::Nvfp4 { .. })).count(),
        "variants": reports,
    }))
}

/// One overlapped drafting round as the worker submits it
/// (`begin_overlapped_verify`), returning the proposals and both commands'
/// timings. The verifier rows stay pending for `commit_round`.
fn overlapped_round(
    session: &mut Session,
    anchor: u32,
    depth: usize,
) -> Result<(Vec<u32>, Timing, Timing)> {
    let rows = depth + 1;
    session.check_append(rows)?;
    let workspace = session.verifier.as_ref().ok_or("Inkling verifier workspace unavailable")?;
    let inputs = [anchor, 0, 0, 0];
    let mut target = None;
    let parallel = session.runtime.parallel_route;
    let matvec = session.runtime.verify_nvfp4_matvec;
    session.runtime.parallel_route = true;
    session.runtime.verify_nvfp4_matvec = true;
    let draft = session.active_dspark().ok_or("Inkling draft unavailable")?;
    let result = draft.propose_many_with_submit(
        &session.runtime,
        anchor,
        session.position,
        &session.embedding,
        &session.embed_norm,
        &session.unembedding,
        depth,
        &workspace.draft,
        || {
            target = Some(execute_prepared(
                &session.runtime,
                None,
                |command| workspace.save(command, session, rows),
                |gpu| {
                    session.encode_verify_transaction(
                        gpu,
                        &workspace.lanes,
                        &inputs[..rows],
                        super::prefill::VerifyExperts::Grouped,
                        workspace,
                        Some(workspace.draft.winners()),
                    )
                },
            )?);
            Ok(())
        },
    );
    session.runtime.verify_nvfp4_matvec = matvec;
    session.runtime.parallel_route = parallel;
    match (result, target) {
        (Ok((proposals, draft)), Some(target)) => Ok((proposals, draft, target)),
        (result, _) => {
            session.reset();
            Err(result.err().unwrap_or("Inkling overlapped round did not submit"))
        },
    }
}

/// `Verification::commit` with its command timing.
fn commit_round(
    session: &mut Session,
    rows: usize,
    kept: usize,
) -> Result<Timing> {
    let workspace = session.verifier.as_ref().ok_or("Inkling verifier workspace unavailable")?;
    let draft = session.active_dspark().ok_or("Inkling draft unavailable")?;
    let position = session.position;
    let timing = execute_prepared(
        &session.runtime,
        None,
        |command| workspace.restore_suffix(command, session, rows, kept),
        |gpu| {
            for (row, taps) in workspace.lanes.verify_taps(session, rows).take(kept).enumerate() {
                draft.commit(gpu, taps, position + row);
            }
        },
    )?;
    session.position += kept;
    session.features_ready = false;
    Ok(timing)
}

fn row_logits(
    session: &Session,
    row: usize,
) -> &[f32] {
    // Read after the verifier command completed; nothing writes it meanwhile.
    unsafe {
        std::slice::from_raw_parts(
            session.logits.contents().as_ptr().cast::<f32>().add(row * PADDED_VOCABULARY),
            VOCABULARY,
        )
    }
}

/// Worker-faithful greedy drafting loop: per-round wall time split into the
/// draft and verifier commands, the commit command, host sampling and idle
/// time, with per-position acceptance, at each depth and for serial steps.
pub fn measure_round_budget(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    outputs: usize,
) -> Result<Value> {
    use std::time::Instant;
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = load_worker_session(checkpoint, Some(draft), context)?;
    if prompt + outputs + MAX_VERIFY_ROWS > context {
        return Err("Inkling round budget exceeds the context");
    }
    ingest(&mut session, &tokens, prompt)?;
    let seed = session.step(tokens[prompt - 1])?;
    let first = transaction_validation::greedy(&seed);
    let base = session.snapshot()?;
    // Serial reference and step timing.
    let mut teacher = vec![first];
    let mut step_ms = Vec::new();
    while teacher.len() < outputs {
        let started = Instant::now();
        let logits = session.step(*teacher.last().unwrap())?;
        let token = transaction_validation::greedy(&logits);
        step_ms.push(1000.0 * started.elapsed().as_secs_f64());
        teacher.push(token);
        if token == END_OF_SAMPLING {
            break;
        }
    }
    // The worker stops at the end-of-sampling token. Rounds whose anchor
    // lies after the first visible-text marker count as answer text.
    let outputs = teacher.len();
    let answer_from = teacher.iter().position(|&token| token == CONTENT_TEXT).unwrap_or(outputs);
    let mut depths = Vec::new();
    for depth in 1..MAX_VERIFY_ROWS {
        session.restore(&base)?;
        let mut output = vec![first];
        let mut reached = [[0usize; MAX_VERIFY_ROWS]; 2];
        let mut accepted = [[0usize; MAX_VERIFY_ROWS]; 2];
        let mut phase_rounds = [0usize; 2];
        let mut phase_tokens = [0usize; 2];
        let mut parts: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let started = Instant::now();
        let mut rounds = 0;
        while output.len() < outputs {
            let anchor = *output.last().unwrap();
            let next_depth = depth.min(outputs - output.len() - 1);
            if next_depth == 0 {
                output.push(transaction_validation::greedy(&session.step(anchor)?));
                continue;
            }
            rounds += 1;
            let phase = usize::from(output.len() - 1 >= answer_from);
            phase_rounds[phase] += 1;
            let round_start = Instant::now();
            let (proposals, draft_timing, verify_timing) = overlapped_round(&mut session, anchor, next_depth)?;
            let submitted = Instant::now();
            let mut kept = 1;
            for row in 0..=proposals.len() {
                let scores = row_logits(&session, row);
                if scores.iter().any(|x| !x.is_finite()) {
                    session.reset();
                    return Err("Inkling round produced nonfinite logits");
                }
                let token = transaction_validation::greedy(scores);
                output.push(token);
                if row < proposals.len() {
                    reached[phase][row] += 1;
                }
                if proposals.get(row) != Some(&token) {
                    break;
                }
                accepted[phase][row] += 1;
                kept += 1;
            }
            phase_tokens[phase] += kept;
            let sampled = Instant::now();
            let commit = commit_round(&mut session, next_depth + 1, kept)?;
            let done = Instant::now();
            let ms = |a: Instant, b: Instant| 1000.0 * (b - a).as_secs_f64();
            parts.entry("round_wall_ms").or_default().push(ms(round_start, done));
            parts.entry("draft_plus_verify_wall_ms").or_default().push(ms(round_start, submitted));
            parts.entry("draft_gpu_ms").or_default().push(draft_timing.gpu_ms);
            parts.entry("verify_gpu_ms").or_default().push(verify_timing.gpu_ms);
            parts.entry("verify_encode_ms").or_default().push(verify_timing.encode_ms);
            parts.entry("draft_encode_ms").or_default().push(draft_timing.encode_ms);
            parts.entry("host_readback_and_sampling_ms").or_default().push(ms(submitted, sampled));
            parts.entry("commit_wall_ms").or_default().push(ms(sampled, done));
            parts.entry("commit_gpu_ms").or_default().push(commit.gpu_ms);
            parts
                .entry("idle_and_host_ms")
                .or_default()
                .push(ms(round_start, done) - draft_timing.gpu_ms - verify_timing.gpu_ms - commit.gpu_ms);
        }
        let elapsed = started.elapsed().as_secs_f64();
        output.truncate(outputs);
        if output != teacher {
            return Err("Inkling round budget output differs from serial decode");
        }
        let summary = parts
            .into_iter()
            .map(|(key, mut values)| (key.to_owned(), json!(median(&mut values))))
            .collect::<serde_json::Map<_, _>>();
        depths.push(json!({
            "depth": depth,
            "rounds": rounds,
            "tokens_per_second": (outputs - 1) as f64 / elapsed,
            "tokens_per_round": (outputs - 1) as f64 / rounds.max(1) as f64,
            "thinking": {"rounds": phase_rounds[0], "tokens": phase_tokens[0],
                "per_position_reached": &reached[0][..depth], "per_position_accepted": &accepted[0][..depth]},
            "answer": {"rounds": phase_rounds[1], "tokens": phase_tokens[1],
                "per_position_reached": &reached[1][..depth], "per_position_accepted": &accepted[1][..depth]},
            "median": summary,
        }));
    }
    // Adaptive depth: the planner chooses 0 to 3 drafts per round from its
    // running acceptance and cost estimates, as the worker does.
    {
        session.restore(&base)?;
        let mut planner = DepthPlanner::new();
        let mut output = vec![first];
        let mut chosen = [0usize; MAX_VERIFY_ROWS];
        let started = Instant::now();
        while output.len() < outputs {
            let anchor = *output.last().unwrap();
            let depth = planner.choose(outputs - output.len() - 1);
            chosen[depth] += 1;
            let round_start = Instant::now();
            if depth == 0 {
                output.push(transaction_validation::greedy(&session.step(anchor)?));
                planner.record(0, 0, 1000.0 * round_start.elapsed().as_secs_f64());
                continue;
            }
            let (proposals, _, _) = overlapped_round(&mut session, anchor, depth)?;
            let mut kept = 1;
            for row in 0..=proposals.len() {
                let token = transaction_validation::greedy(row_logits(&session, row));
                output.push(token);
                if proposals.get(row) != Some(&token) {
                    break;
                }
                kept += 1;
            }
            commit_round(&mut session, depth + 1, kept)?;
            planner.record(depth, kept - 1, 1000.0 * round_start.elapsed().as_secs_f64());
        }
        let elapsed = started.elapsed().as_secs_f64();
        output.truncate(outputs);
        if output != teacher {
            return Err("Inkling adaptive depth output differs from serial decode");
        }
        depths.push(json!({
            "depth": "adaptive",
            "tokens_per_second": (outputs - 1) as f64 / elapsed,
            "rounds_by_depth": chosen,
        }));
    }
    session.restore(&base)?;
    Ok(json!({
        "prompt_tokens": prompt,
        "outputs": outputs,
        "thinking_tokens": answer_from.min(outputs),
        "affine8": session.runtime.affine8.is_some(),
        "serial_step_wall_ms_median": median(&mut step_ms),
        "serial_tokens_per_second": 1000.0 / median(&mut step_ms.clone()),
        "depths": depths,
    }))
}

/// Diagnostic controls that switch one decode change off in a loaded session.
pub(super) fn set_control(
    session: &mut Session,
    control: &str,
    enabled: bool,
) -> Result<()> {
    match control {
        "short_attention" => {
            session.verifier.as_mut().ok_or("Inkling verifier workspace unavailable")?.short_attention = enabled;
        },
        "router" => {
            // Swap the decode router pipeline for the previous MPP kernel.
            const SAVED: Kernel = "ink_router_decode_saved";
            const NATIVE: Kernel = "ink_verify_router_native";
            const REFERENCE: Kernel = "ink_verify_router_mpp";
            let pipelines = &mut session.runtime.pipelines;
            if !pipelines.contains_key(SAVED) {
                let current = pipelines.get(NATIVE).ok_or("Inkling decode router unavailable")?.clone();
                pipelines.insert(SAVED, current);
                affine8::compile(&mut session.runtime, include_str!("router_mpp_reference.metal"), &[REFERENCE])?;
            }
            let pipelines = &mut session.runtime.pipelines;
            let chosen = pipelines[if enabled {
                SAVED
            } else {
                REFERENCE
            }]
            .clone();
            pipelines.insert(NATIVE, chosen);
        },
        "expert_inputs" => {
            // Off flips the NVFP4 expert input policy for single-row decode
            // and the verifier (prefill keeps the compiled policy).
            const SINGLE: Kernel = "ink_nvfp4_activation";
            const VERIFY: Kernel = "ink_verify_half_activation";
            const SAVED_SINGLE: Kernel = "ink_expert_inputs_saved_single";
            const SAVED_VERIFY: Kernel = "ink_expert_inputs_saved_verify";
            const FLIP_SINGLE: Kernel = "ink_expert_inputs_flip_single";
            const FLIP_VERIFY: Kernel = "ink_expert_inputs_flip_verify";
            if !session.runtime.pipelines.contains_key(SAVED_SINGLE) {
                let pipelines = &mut session.runtime.pipelines;
                let single = pipelines.get(SINGLE).ok_or("Inkling activation kernel unavailable")?.clone();
                pipelines.insert(SAVED_SINGLE, single);
                // Sessions without the verifier have no half activation kernel.
                if let Some(verify) = pipelines.get(VERIFY).cloned() {
                    pipelines.insert(SAVED_VERIFY, verify);
                }
                let flip = if QUANTIZE_EXPERT_INPUTS {
                    "0"
                } else {
                    "1"
                };
                let single_source = format!(
                    "#include <metal_stdlib>\nusing namespace metal;\n#define INK_EXPERT_INPUTS_QUANTIZED {flip}\n#define ink_nvfp4_activation {FLIP_SINGLE}\n{}",
                    include_str!("experts.metal")
                );
                affine8::compile(&mut session.runtime, &single_source, &[FLIP_SINGLE])?;
                let verify_source = format!(
                    "#include <metal_stdlib>\nusing namespace metal;\n#define INK_EXPERT_INPUTS_QUANTIZED {flip}\n#define INK_VERIFY_HALF_ACT\n#define ink_nvfp4_activation {FLIP_VERIFY}\n{}",
                    include_str!("experts.metal")
                );
                affine8::compile(&mut session.runtime, &verify_source, &[FLIP_VERIFY])?;
            }
            let pipelines = &mut session.runtime.pipelines;
            let (single, verify) = if enabled {
                (SAVED_SINGLE, SAVED_VERIFY)
            } else {
                (FLIP_SINGLE, FLIP_VERIFY)
            };
            let single = pipelines[single].clone();
            pipelines.insert(SINGLE, single);
            if pipelines.contains_key(SAVED_VERIFY) {
                let verify = pipelines[verify].clone();
                pipelines.insert(VERIFY, verify);
            }
            // The direct products assume unquantized inputs.
            if !enabled {
                session.runtime.direct_expert_inputs = false;
            }
        },
        "direct_expert_inputs" => {
            // On skips the activation passes before the single-row and
            // verifier products (diagnostic; off is the default).
            let available = !QUANTIZE_EXPERT_INPUTS
                && super::DIRECT_PRODUCTS.iter().all(|name| session.runtime.pipelines.contains_key(name));
            session.runtime.direct_expert_inputs = enabled && available;
        },
        _ => return Err("unknown Inkling decode control"),
    }
    Ok(())
}

/// Paired ABBA timing of one decode change in one loaded session: the
/// overlapped round (draft and verifier commands) at depths 1 to 3 and a
/// single step, with the change on (candidate) and off (base).
pub fn measure_paired_control(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    pairs: usize,
    control: &str,
) -> Result<Value> {
    use std::time::Instant;
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = load_worker_session(checkpoint, Some(draft), context)?;
    ingest(&mut session, &tokens, prompt)?;
    let anchor = transaction_validation::greedy(&session.step(tokens[prompt - 1])?);
    let base = session.snapshot()?;
    // Logit agreement with the control on and off: the single step and every
    // row of a full-depth verifier round from the same state.
    let mut sides = Vec::new();
    for enabled in [false, true] {
        set_control(&mut session, control, enabled)?;
        session.restore(&base)?;
        let mut logits = vec![session.step(anchor)?];
        session.restore(&base)?;
        overlapped_round(&mut session, anchor, MAX_VERIFY_ROWS - 1)?;
        logits.extend((0..MAX_VERIFY_ROWS).map(|row| row_logits(&session, row).to_vec()));
        session.reset();
        sides.push(logits);
    }
    let agreement = sides[0]
        .iter()
        .zip(&sides[1])
        .enumerate()
        .map(|(index, (off, on))| {
            let unequal = off.iter().zip(on).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let max_abs = off.iter().zip(on).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            json!({
                "row": if index == 0 { "step".to_string() } else { format!("verify{}", index - 1) },
                "values": off.len(), "unequal": unequal, "max_abs": max_abs,
            })
        })
        .collect::<Vec<_>>();
    let mut verify_gpu = vec![Vec::new(); MAX_VERIFY_ROWS];
    let mut round_wall = vec![Vec::new(); MAX_VERIFY_ROWS];
    let mut step_gpu = Vec::new();
    for pair in 0..pairs {
        let order = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut gpu = [[0.0; 2]; MAX_VERIFY_ROWS];
        let mut wall = [[0.0; 2]; MAX_VERIFY_ROWS];
        let mut step = [0.0; 2];
        for enabled in order {
            set_control(&mut session, control, enabled)?;
            let slot = usize::from(enabled);
            session.restore(&base)?;
            let result = session.encode_append(&[anchor], true, None);
            let gpu_ms = result.as_ref().map(|t| t.gpu_ms).map_err(|e| *e);
            session.finish_append(result, 1)?;
            step[slot] = gpu_ms?;
            for depth in 1..MAX_VERIFY_ROWS {
                session.restore(&base)?;
                let started = Instant::now();
                let (_, _, verify) = overlapped_round(&mut session, anchor, depth)?;
                wall[depth][slot] = 1000.0 * started.elapsed().as_secs_f64();
                gpu[depth][slot] = verify.gpu_ms;
                session.reset();
            }
        }
        step_gpu.push((step[0], step[1]));
        for depth in 1..MAX_VERIFY_ROWS {
            verify_gpu[depth].push((gpu[depth][0], gpu[depth][1]));
            round_wall[depth].push((wall[depth][0], wall[depth][1]));
        }
    }
    set_control(&mut session, control, true)?;
    let summary = super::affine8_validation::timing_summary;
    Ok(json!({
        "prompt_tokens": prompt,
        "control": control,
        "note": "base = control off, candidate = control on (labels bf16/affine8 in each summary mean base/candidate)",
        "logit_agreement": agreement,
        "single_step_gpu_ms": summary(&step_gpu),
        "verify_gpu_ms": (1..MAX_VERIFY_ROWS).map(|d| json!({"rows": d + 1, "timing": summary(&verify_gpu[d])})).collect::<Vec<_>>(),
        "draft_plus_verify_wall_ms": (1..MAX_VERIFY_ROWS).map(|d| json!({"rows": d + 1, "timing": summary(&round_wall[d])})).collect::<Vec<_>>(),
    }))
}

/// Draft acceptance trace for depth-policy simulation: at every anchor of the
/// greedy output, the number of leading proposals of a depth-3 round that the
/// target accepts (0 to 3), plus round and step wall times measured
/// interleaved at the same positions. Only counts and times are reported.
pub fn measure_draft_trace(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    outputs: usize,
) -> Result<Value> {
    use std::time::Instant;
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = load_worker_session(checkpoint, Some(draft), context)?;
    if prompt + outputs + MAX_VERIFY_ROWS > context {
        return Err("Inkling draft trace exceeds the context");
    }
    ingest(&mut session, &tokens, prompt)?;
    let mut anchor = transaction_validation::greedy(&session.step(tokens[prompt - 1])?);
    let mut lengths = Vec::with_capacity(outputs);
    let mut confidences: Vec<Vec<f32>> = Vec::with_capacity(outputs);
    let mut costs = vec![Vec::new(); MAX_VERIFY_ROWS];
    // Non-overlapped confidence rounds: the proposal command, then a verify
    // of the chosen rows submitted after the host has read the confidences.
    let mut propose_costs = Vec::new();
    let mut plain_verify = vec![Vec::new(); MAX_VERIFY_ROWS];
    for index in 0..outputs {
        if anchor == END_OF_SAMPLING {
            break;
        }
        let snapshot = session.snapshot()?;
        // Round costs at every depth every eighth position, in rotating order.
        if index % 8 == 0 {
            for offset in 0..MAX_VERIFY_ROWS {
                let depth = (index / 8 + offset) % MAX_VERIFY_ROWS;
                session.restore(&snapshot)?;
                let started = Instant::now();
                if depth == 0 {
                    session.step(anchor)?;
                } else {
                    let _ = overlapped_round(&mut session, anchor, depth)?;
                    session.reset();
                }
                costs[depth].push(1000.0 * started.elapsed().as_secs_f64());
            }
            session.restore(&snapshot)?;
            let started = Instant::now();
            let proposals = session.propose_verify(anchor, MAX_VERIFY_ROWS - 1)?;
            propose_costs.push(1000.0 * started.elapsed().as_secs_f64());
            for depth in 1..MAX_VERIFY_ROWS {
                session.restore(&snapshot)?;
                let inputs = std::iter::once(anchor).chain(proposals.iter().copied().take(depth)).collect::<Vec<_>>();
                let started = Instant::now();
                let txn = session.begin_verify(&inputs)?;
                txn.logits(depth)?;
                plain_verify[depth].push(1000.0 * started.elapsed().as_secs_f64());
                drop(txn);
            }
            session.restore(&snapshot)?;
        }
        let (proposals, _, _) = overlapped_round(&mut session, anchor, MAX_VERIFY_ROWS - 1)?;
        confidences.push(
            session
                .verifier
                .as_ref()
                .ok_or("Inkling verifier workspace unavailable")?
                .draft
                .confidences(proposals.len()),
        );
        let mut accepted = 0;
        let mut first = None;
        for (row, &proposal) in proposals.iter().enumerate() {
            let token = transaction_validation::greedy(row_logits(&session, row));
            if row == 0 {
                first = Some(token);
            }
            if token != proposal {
                break;
            }
            accepted += 1;
        }
        if proposals.is_empty() {
            first = Some(transaction_validation::greedy(row_logits(&session, 0)));
        }
        session.reset();
        session.restore(&snapshot)?;
        let next = transaction_validation::greedy(&session.step(anchor)?);
        if Some(next) != first {
            return Err("Inkling draft trace verifier row differs from the serial step");
        }
        lengths.push(accepted);
        anchor = next;
    }
    let medians = costs.iter_mut().map(|values| median(values)).collect::<Vec<_>>();
    let propose = median(&mut propose_costs);
    let verify = plain_verify.iter_mut().map(|values| median(values)).collect::<Vec<_>>();
    // Confidence-gated costs: draft then step, or draft then a d-row verify.
    let gated = (0..MAX_VERIFY_ROWS)
        .map(|depth| {
            propose
                + if depth == 0 {
                    medians[0]
                } else {
                    verify[depth]
                }
        })
        .collect::<Vec<_>>();
    // Policy simulation on the recorded trace. A round either runs the
    // overlapped draft and verify at a chosen depth (costs medians[depth]),
    // a plain step without a draft (medians[0]), or a confidence-gated round:
    // the proposal command, then a verify of the confident prefix or a plain
    // step (costs gated[depth]).
    #[derive(Clone, Copy)]
    enum Mode {
        Fixed(usize),
        Planner(bool),
        Confidence(f32),
    }
    let simulate = |mode: Mode, backoff: bool| {
        let mut planner = match mode {
            Mode::Planner(true) => DepthPlanner::with_plain_steps(),
            _ => DepthPlanner::new(),
        };
        let (mut position, mut produced, mut time) = (0usize, 0usize, 0.0f64);
        let (mut off, mut skip) = (0usize, 0usize);
        let mut warmup: usize = if backoff {
            4
        } else {
            0
        };
        while position < lengths.len() {
            let limit = (MAX_VERIFY_ROWS - 1).min(lengths.len() - position);
            if warmup > 0 || skip > 0 || limit == 0 {
                warmup = warmup.saturating_sub(1);
                skip = skip.saturating_sub(1);
                produced += 1;
                time += medians[0];
                position += 1;
                continue;
            }
            let (depth, cost) = match mode {
                Mode::Fixed(depth) => (depth.min(limit), medians[depth.min(limit)]),
                Mode::Planner(_) => {
                    let depth = planner.choose(limit);
                    (depth, medians[depth])
                },
                Mode::Confidence(threshold) => {
                    let mut product = 1.0f32;
                    let mut depth = 0;
                    for &value in confidences[position].iter().take(limit) {
                        product *= value;
                        if product < threshold {
                            break;
                        }
                        depth += 1;
                    }
                    (depth, gated[depth])
                },
            };
            let accepted = lengths[position].min(depth);
            produced += accepted + 1;
            time += cost;
            planner.record(depth, accepted, cost);
            position += accepted + 1;
            if backoff {
                // Drafting pays when the tokens it gains, at the plain step's
                // price, cover what the round cost.
                if (accepted + 1) as f64 * medians[0] < cost {
                    off = (2 * off).clamp(1, 32);
                    skip = off;
                } else {
                    off = 0;
                }
            }
        }
        1000.0 * produced as f64 / time
    };
    let fixed = (0..MAX_VERIFY_ROWS).map(|depth| simulate(Mode::Fixed(depth), false)).collect::<Vec<_>>();
    let fixed_backoff = (1..MAX_VERIFY_ROWS).map(|depth| simulate(Mode::Fixed(depth), true)).collect::<Vec<_>>();
    let history = [simulate(Mode::Planner(false), false), simulate(Mode::Planner(true), false)];
    let history_backoff = [simulate(Mode::Planner(false), true), simulate(Mode::Planner(true), true)];
    let mut confidence = Vec::new();
    for step in 2..=18 {
        let threshold = step as f32 * 0.05;
        confidence.push(json!({
            "threshold": threshold,
            "tokens_per_second": simulate(Mode::Confidence(threshold), false),
            "with_backoff": simulate(Mode::Confidence(threshold), true),
        }));
    }
    Ok(json!({
        "simulated_tokens_per_second": {
            "fixed_0_to_3": fixed,
            "fixed_1_to_3_with_backoff": fixed_backoff,
            "history_planner_floor1_floor0": history,
            "history_planner_with_backoff": history_backoff,
            "confidence_gated": confidence,
        },
        "propose_ms_median": propose,
        "plain_verify_ms_median_by_depth": verify,
        "confidence_gated_cost_ms_by_depth": gated,
        "prompt_tokens": prompt,
        "affine8": session.runtime.affine8.is_some(),
        "positions": lengths.len(),
        "accept_lengths": lengths,
        "cost_ms_median_by_depth": medians,
        "cost_samples": costs.first().map_or(0, Vec::len),
    }))
}

/// Generate the greedy continuation from the base snapshot with a fixed
/// depth or, when none is given, the depth planner. Returns tokens and wall
/// seconds.
fn generate_with_policy(
    session: &mut Session,
    base: &Snapshot,
    first: u32,
    outputs: usize,
    fixed: Option<usize>,
) -> Result<(Vec<u32>, f64)> {
    use std::time::Instant;
    session.restore(base)?;
    let mut planner = DepthPlanner::new();
    let mut output = vec![first];
    let started = Instant::now();
    while output.len() < outputs && *output.last().unwrap() != END_OF_SAMPLING {
        let anchor = *output.last().unwrap();
        let limit = (MAX_VERIFY_ROWS - 1).min(outputs - output.len() - 1);
        let depth = fixed.map_or_else(|| planner.choose(limit), |depth| depth.min(limit));
        let round_start = Instant::now();
        if depth == 0 {
            output.push(transaction_validation::greedy(&session.step(anchor)?));
            planner.record(0, 0, 1000.0 * round_start.elapsed().as_secs_f64());
            continue;
        }
        let (proposals, _, _) = overlapped_round(session, anchor, depth)?;
        let mut kept = 1;
        for row in 0..=proposals.len() {
            let token = transaction_validation::greedy(row_logits(session, row));
            output.push(token);
            if proposals.get(row) != Some(&token) {
                break;
            }
            kept += 1;
        }
        commit_round(session, depth + 1, kept)?;
        planner.record(depth, kept - 1, 1000.0 * round_start.elapsed().as_secs_f64());
    }
    let seconds = started.elapsed().as_secs_f64();
    // A round can commit tokens past the end of sampling; the worker stops there.
    if let Some(end) = output.iter().position(|&token| token == END_OF_SAMPLING) {
        output.truncate(end + 1);
    }
    output.truncate(outputs);
    Ok((output, seconds))
}

/// Fixed depths 1 to 3 and the planner, interleaved in rotating order within
/// one loaded session; every arm must reproduce the serial greedy output.
pub fn measure_depth_policies(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    outputs: usize,
    repeats: usize,
) -> Result<Value> {
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prompt + 3])?;
    let mut session = load_worker_session(checkpoint, Some(draft), context)?;
    ingest(&mut session, &tokens, prompt)?;
    let first = transaction_validation::greedy(&session.step(tokens[prompt - 1])?);
    let base = session.snapshot()?;
    let (teacher, _) = generate_with_policy(&mut session, &base, first, outputs, Some(0))?;
    let arms: [Option<usize>; 4] = [Some(1), Some(2), Some(3), None];
    let mut rates = vec![Vec::new(); arms.len()];
    for repeat in 0..repeats {
        let mut order = (0..arms.len()).map(|i| (i + repeat) % arms.len()).collect::<Vec<_>>();
        if repeat % 2 == 1 {
            order.reverse();
        }
        for arm in order {
            let (output, seconds) = generate_with_policy(&mut session, &base, first, outputs, arms[arm])?;
            if output != teacher {
                return Err("Inkling depth policy changed the greedy output");
            }
            rates[arm].push((output.len() - 1) as f64 / seconds);
        }
    }
    let names = ["d1", "d2", "d3", "adaptive"];
    let mut report = serde_json::Map::new();
    for (name, values) in names.iter().zip(&mut rates) {
        report.insert((*name).into(), json!({"tokens_per_second_median": median(values), "samples": values.len()}));
    }
    session.restore(&base)?;
    Ok(
        json!({"prompt_tokens": prompt, "outputs": teacher.len(), "affine8": session.runtime.affine8.is_some(), "arms": report}),
    )
}
