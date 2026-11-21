// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only full-target check against a private pinned-runtime capture.

use std::time::Instant;

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

fn report(
    got: &[f32],
    want: &[f32],
) -> Value {
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    let mut peak = 0.0f64;
    let mut peak_error = 0.0f64;
    let mut unequal = 0;
    for (&got, &want) in got.iter().zip(want) {
        let delta = f64::from(got) - f64::from(want);
        signal += f64::from(want).powi(2);
        noise += delta * delta;
        peak = peak.max(f64::from(want).abs());
        peak_error = peak_error.max(delta.abs());
        unequal += usize::from(got.to_bits() != want.to_bits());
    }
    json!({
        "values": got.len(),
        "unequal": unequal,
        "relative_l2": noise.sqrt() / signal.sqrt().max(f64::MIN_POSITIVE),
        "peak_relative_error": peak_error / peak.max(f64::MIN_POSITIVE),
    })
}

fn oracle_bf16(
    source: &Tensors,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    Ok(source.values::<u16>(name, "BF16", shape)?.into_iter().map(|bits| bf16::from_bits(bits).to_f32()).collect())
}

fn as_float(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&word| bf16::from_bits(word).to_f32()).collect()
}

fn native_digest<T: bytemuck::Pod>(values: &[T]) -> String {
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

fn state_digest(session: &Session) -> String {
    let mut digest = Sha256::new();
    digest.update((session.position as u64).to_le_bytes());
    for visit in &session.visits {
        let kv_bytes = session.position.min(visit.capacity) * KV * 2;
        let state = [(&visit.keys, kv_bytes), (&visit.values, kv_bytes)]
            .into_iter()
            .chain(visit.conv.iter().map(|conv| (conv, conv.length() * session.position.min(3) / 3)));
        for (buffer, bytes) in state {
            // The previous command has completed. Hash only committed slots in
            // the shared buffer and never copy their values into the report.
            digest.update(unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), bytes) });
        }
    }
    format!("{:x}", digest.finalize())
}

fn probe_partials(
    session: &Session,
    source: &Tensors,
) -> Result<Vec<Value>> {
    session
        .visits
        .iter()
        .enumerate()
        .map(|(index, visit)| {
            let got = read_shared::<f32>(&visit.scratch.partial_debug, WIDTH);
            let want = oracle_bf16(source, &format!("full.visit.{index:02}.partial"), &[1, 1, WIDTH])?;
            Ok(report(&got, &want))
        })
        .collect()
}

fn probe_history(
    session: &Session,
    source: &Tensors,
    physical: usize,
) -> Result<Value> {
    let prefix = 100 + physical;
    if !source.contains(&format!("{prefix}.history.key")) {
        return Ok(Value::Null);
    }
    let visit = &session.visits[physical];
    let order = (0..3).map(|lag| (session.position + lag) % 3);
    let mut values = serde_json::Map::new();
    for (kind, buffer, width) in [
        ("key", &visit.conv[0], KV),
        ("value", &visit.conv[1], KV),
        ("attn", &visit.conv[2], WIDTH),
        ("mlp", &visit.conv[3], WIDTH),
    ] {
        let want = oracle_bf16(source, &format!("{prefix}.history.{kind}"), &[1, 3, width])?;
        let raw = read_shared::<u16>(buffer, 3 * width);
        let mut got = vec![0.0; 3 * width];
        for (i, slot) in order.clone().enumerate().skip(3usize.saturating_sub(session.position)) {
            got[i * width..(i + 1) * width].copy_from_slice(&as_float(&raw[slot * width..(slot + 1) * width]));
        }
        values.insert(kind.into(), report(&got, &want));
    }
    Ok(Value::Object(values))
}

fn probe_first_pass(
    session: &Session,
    source: &Tensors,
    physical: usize,
) -> Result<Value> {
    let visit = &session.visits[physical];
    let s = &visit.scratch;
    let prefix = format!("{}.", 100 + physical);
    if !source.contains(&format!("{prefix}trace.query_clipped")) {
        return Ok(Value::Null);
    }
    let mut values = serde_json::Map::new();
    let stages: &[(&str, &Buffer, &[usize])] = &[
        ("trace.attn_norm_0", &s.qkv_input[0], &[1, 1, WIDTH]),
        ("trace.attn_norm_1", &s.qkv_input[1], &[1, 1, WIDTH]),
        ("trace.attn_norm_2", &s.qkv_input[2], &[1, 1, WIDTH]),
        ("trace.query_clipped", &s.q, &[1, HEADS, 1, 128]),
        ("trace.key_raw", &s.k, &[1, 1, KV]),
        ("trace.value_raw", &s.v, &[1, 1, KV]),
        ("trace.key_dynamic", &s.dynamic[0], &[1, 1, 8, 4]),
        ("trace.value_dynamic", &s.dynamic[1], &[1, 1, 8, 4]),
        ("trace.key_conv", &s.conv_k, &[1, 1, 8, 128]),
        ("trace.value_conv", &s.conv_v, &[1, 1, 8, 128]),
        ("trace.relative_clipped", &s.r, &[1, 1, HEADS, 16]),
        ("trace.relative_table", &s.table, &[1, HEADS, 1, LOCAL]),
        ("trace.gate_raw", &s.attn_gate, &[1, 1, HEADS * 128]),
        ("trace.sink_raw", &s.sink, &[1, HEADS, 1, 1]),
        ("trace.attended", &s.attended, &[1, 1, HEADS * 128]),
        ("trace.attention_raw", &s.attn_raw, &[1, 1, WIDTH]),
        ("attention", &s.attention, &[1, 1, WIDTH]),
        ("mlp_input", &s.mlp_input, &[1, 1, WIDTH]),
        ("experts", &s.experts, &[1, 1, WIDTH]),
        ("trace.mlp_conv", &s.mlp, &[1, 1, WIDTH]),
    ];
    for &(name, buffer, shape) in stages {
        if !source.contains(&format!("{prefix}{name}")) {
            continue;
        }
        let want = oracle_bf16(source, &format!("{prefix}{name}"), shape)?;
        let got = as_float(&read_shared::<u16>(buffer, want.len()));
        values.insert(name.into(), report(&got, &want));
    }
    let position = session.position - 1;
    let offset = position % visit.capacity * KV;
    for (name, buffer) in [("key_rows", &visit.keys), ("value_rows", &visit.values)] {
        let want = oracle_bf16(source, &format!("{prefix}{name}"), &[1, KV])?;
        // The command has completed, and the cache is shared and contains
        // this validated position.
        let got =
            as_float(unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u16>().add(offset), KV) });
        values.insert(name.into(), report(&got, &want));
    }
    let logits = source.values::<f32>(&format!("{prefix}trace.router_logits"), "F32", &[1, EXPERTS])?;
    values.insert("trace.router_logits".into(), report(&read_shared::<f32>(&s.logits, EXPERTS), &logits));
    let routes = source.values::<u32>(&format!("{prefix}routes"), "U32", &[1, 8])?;
    let got_routes = read_shared::<u32>(&s.routes, 8);
    let aligned = routes.iter().map(|id| got_routes.iter().position(|got| got == id)).collect::<Option<Vec<_>>>();
    values.insert("expert_ids_match".into(), Value::Bool(aligned.is_some()));
    if let Some(indices) = aligned {
        let got_weights = read_shared::<f32>(&s.route_weights, 8);
        let weights = source.values::<f32>(&format!("{prefix}route_weights"), "F32", &[1, 8])?;
        let aligned_weights = indices.iter().map(|&i| got_weights[i]).collect::<Vec<_>>();
        values.insert("route_weights".into(), report(&aligned_weights, &weights));
        let got_mid = read_shared::<u16>(&s.mid, 8 * MID);
        let aligned_mid = indices.iter().flat_map(|&i| as_float(&got_mid[i * MID..(i + 1) * MID])).collect::<Vec<_>>();
        let mid = oracle_bf16(source, &format!("{prefix}mid"), &[1, 8, 1, MID])?;
        values.insert("mid".into(), report(&aligned_mid, &mid));
        let got_down = read_shared::<f32>(&s.down_inner, 8 * MID);
        let aligned_down =
            indices.iter().flat_map(|&i| got_down[i * MID..(i + 1) * MID].iter().copied()).collect::<Vec<_>>();
        let down = source.values::<f32>(&format!("{prefix}down_inner"), "F32", &[1, 8, 1, MID])?;
        values.insert("down_quantized".into(), report(&aligned_down, &down));
    }
    let up = source.values::<f32>(&format!("{prefix}up_inner"), "F32", &[1, 1, 1, WIDTH])?;
    values.insert("up_quantized".into(), report(&read_shared::<f32>(&s.up_inner, WIDTH), &up));
    Ok(Value::Object(values))
}

fn probe_epilogue(
    session: &Session,
    source: &Tensors,
) -> Result<Vec<Value>> {
    let mut outputs = Vec::with_capacity(4);
    for layer in 20..24 {
        let visit = &session.visits[44 + layer - 20];
        let scratch = &visit.scratch;
        let name = |suffix| format!("{layer}.{suffix}");
        let input = source.values::<u16>(&name("mlp_input"), "BF16", &[1, 1, WIDTH])?;
        // The preceding command has completed and this buffer is only read by
        // the upcoming expert probe. Inputs and results never leave this process.
        unsafe { scratch.mlp_input.contents().as_ptr().cast::<u16>().copy_from_nonoverlapping(input.as_ptr(), WIDTH) };
        execute(&session.runtime, |gpu| session.layers[layer].experts(gpu, scratch))?;
        let want = oracle_bf16(source, &name("experts"), &[1, 1, WIDTH])?;
        let got = read_shared::<u16>(&scratch.experts, WIDTH)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let want_inner = source.values::<f32>(&name("up_inner"), "F32", &[1, 1, 1, WIDTH])?;
        let got_inner = read_shared::<f32>(&scratch.up_inner, WIDTH);
        outputs.push(
            json!({"physical": layer, "output": report(&got, &want), "up_inner": report(&got_inner, &want_inner)}),
        );
    }
    Ok(outputs)
}

/// Compare the full target on the four singleton positions in a private
/// pinned-runtime capture. The loader does not infer artifact qualification.
pub fn measure_full(
    checkpoint: &Path,
    oracle: &Path,
    context: usize,
    batched_prefill: bool,
) -> Result<Value> {
    let first = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let prefix = first.values::<u32>("start", "U32", &[1])?[0] as usize;
    if prefix == 0 || prefix.checked_add(4).is_none_or(|len| len > context) {
        return Err("full oracle prefix is outside the context");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prefix + 4])?;
    let started = Instant::now();
    let mut session = Session::load(checkpoint, context)?;
    let serial_allocated = session.runtime.device.current_allocated_size();
    session.set_batched_prefill(batched_prefill)?;
    let prefill_allocated = session.runtime.device.current_allocated_size().saturating_sub(serial_allocated);
    let load_ms = 1000.0 * started.elapsed().as_secs_f64();
    let started = Instant::now();
    session.ingest_many(&tokens[..prefix])?;
    let prefill_ms = 1000.0 * started.elapsed().as_secs_f64();
    let prefill_state = state_digest(&session);
    session.capture_draft_features(true);
    session.capture_visits = true;
    let snapshot = session.snapshot()?;
    let mut rows = Vec::with_capacity(4);
    let mut initial_logits = None;
    for (row, &token) in tokens[prefix..].iter().enumerate() {
        let path = oracle.join(format!("row-{row}.safetensors"));
        let source = Tensors::open(&path)?;
        let history_5 = probe_history(&session, &source, 5)?;
        let history_11 = probe_history(&session, &source, 11)?;
        let started = Instant::now();
        let got = session.step(token)?;
        let elapsed_ms = 1000.0 * started.elapsed().as_secs_f64();
        let want = source.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
        let got_hidden = read_shared::<u16>(&session.hidden, WIDTH)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let want_hidden = oracle_bf16(&source, "full.hidden", &[1, 1, WIDTH])?;
        let top =
            |values: &[f32]| values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index);
        rows.push(json!({
            "milliseconds": elapsed_ms,
            "top_token_matches": top(&got) == top(&want),
            "native_logits_sha256": native_digest(&got),
            "native_taps_sha256": native_digest(&session.draft_features().ok_or("missing full target taps")?),
            "logits": report(&got, &want),
            "hidden": report(&got_hidden, &want_hidden),
            "partial_state": probe_partials(&session, &source)?,
            "history_physical_5": history_5,
            "history_physical_11": history_11,
            "first_pass_physical_4": probe_first_pass(&session, &source, 4)?,
            "first_pass_physical_5": probe_first_pass(&session, &source, 5)?,
            "first_pass_physical_10": probe_first_pass(&session, &source, 10)?,
            "first_pass_physical_11": probe_first_pass(&session, &source, 11)?,
            "isolated_epilogue": probe_epilogue(&session, &source)?,
        }));
        if row == 0 {
            initial_logits = Some(got);
        }
    }
    session.restore(&snapshot)?;
    let after_restore = session.step(tokens[prefix])?;
    let exact_restore = after_restore
        .iter()
        .zip(initial_logits.as_ref().ok_or("empty oracle")?)
        .all(|(restored, original)| restored.to_bits() == original.to_bits());
    session.capture_draft_features(false);
    let late_tile = if batched_prefill && prefix >= LOCAL && session.context - prefix >= BATCHED_PREFILL_ROWS + 1 {
        let tail = &tokens[prefix - BATCHED_PREFILL_ROWS..prefix];
        Some(measure_late_tile(&mut session, &snapshot, tail, tokens[prefix])?)
    } else {
        None
    };
    let mut prefix_rows = Vec::new();
    if prefix == 4 && oracle.join("prefix-0.safetensors").exists() {
        session.reset();
        for (row, &token) in tokens[..prefix].iter().enumerate() {
            let source = Tensors::open(&oracle.join(format!("prefix-{row}.safetensors")))?;
            let history_5 = probe_history(&session, &source, 5)?;
            let history_11 = probe_history(&session, &source, 11)?;
            session.ingest(token)?;
            prefix_rows.push(json!({
                "partial_state": probe_partials(&session, &source)?,
                "history_physical_5": history_5,
                "history_physical_11": history_11,
                "first_pass_physical_4": probe_first_pass(&session, &source, 4)?,
                "first_pass_physical_5": probe_first_pass(&session, &source, 5)?,
                "first_pass_physical_10": probe_first_pass(&session, &source, 10)?,
                "first_pass_physical_11": probe_first_pass(&session, &source, 11)?,
            }));
        }
    }
    Ok(json!({
        "load_ms": load_ms,
        "prefill_input_tokens": prefix,
        "prefill_command_rows": session.prefill_batch_rows(),
        "prefill_additional_metal_allocated_bytes": prefill_allocated,
        "prefill_native_cache_sha256": prefill_state,
        "prefill_ms": prefill_ms,
        "exact_snapshot_replay": exact_restore,
        "prefix_rows": prefix_rows,
        "late_tile": late_tile,
        "rows": rows,
    }))
}

fn measure_late_tile(
    session: &mut Session,
    snapshot: &Snapshot,
    tokens: &[u32],
    next: u32,
) -> Result<Value> {
    let mut run = |layer_major: bool| {
        session.restore(snapshot)?;
        // This diagnostic deliberately holds the single starting state fixed
        // while comparing schedules. Keep the public restore's mode check.
        let mut candidate = None;
        if !layer_major {
            std::mem::swap(&mut candidate, &mut session.prefill_lanes);
        }
        let started = Instant::now();
        session.ingest_many(tokens)?;
        let ms = 1000.0 * started.elapsed().as_secs_f64();
        let state = state_digest(session);
        let logits = native_digest(&session.step(next)?);
        if !layer_major {
            std::mem::swap(&mut candidate, &mut session.prefill_lanes);
        }
        Ok::<_, &'static str>((ms, state, logits))
    };
    let _ = run(false)?;
    let _ = run(true)?;
    let mut exact = true;
    let mut ratios = Vec::new();
    for pair in 0..4 {
        let candidate_first = pair % 2 != 0;
        let first = run(candidate_first)?;
        let second = run(!candidate_first)?;
        let (base, candidate) = if candidate_first {
            (second, first)
        } else {
            (first, second)
        };
        exact &= base.1 == candidate.1 && base.2 == candidate.2;
        ratios.push(candidate.0 / base.0);
    }
    Ok(json!({"rows": tokens.len(), "exact_state_and_logits": exact, "paired_wall_ratios": ratios}))
}

/// Budget the target at the same controlled 4K cache depth as the slice
/// report. This deliberately leaves sampler, draft and worker I/O outside
/// the timed target budget.
pub fn measure_full_budget(
    checkpoint: &Path,
    seed_path: &Path,
) -> Result<Value> {
    const WARMUP: usize = 8;
    const SAMPLES: usize = 32;
    const SYNTHETIC_TOKENS: [u32; 8] = [15, 407, 866, 1520, 3400, 5220, 8800, 12017];
    let started = Instant::now();
    let mut session = Session::load(checkpoint, CAPACITY)?;
    let load_ms = 1000.0 * started.elapsed().as_secs_f64();
    let seed = Tensors::open(seed_path)?;
    for visit in &session.visits {
        validation::seed_visit(&seed, visit)?;
    }
    session.position = START;
    let mut wall = Vec::with_capacity(SAMPLES);
    let mut gpu_ms = Vec::with_capacity(SAMPLES);
    let mut dispatches = 0;
    for index in 0..WARMUP + SAMPLES {
        let token = SYNTHETIC_TOKENS[index % SYNTHETIC_TOKENS.len()];
        let started = Instant::now();
        let timing = execute(&session.runtime, |gpu| {
            session.encode_token(gpu, TokenInput::Text(token), session.position, true, false);
        })?;
        session.position += 1;
        session.read_logits()?;
        if index >= WARMUP {
            wall.push(1000.0 * started.elapsed().as_secs_f64());
            gpu_ms.push(timing.gpu_ms);
            dispatches = timing.dispatches;
        }
    }
    wall.sort_unstable_by(f64::total_cmp);
    gpu_ms.sort_unstable_by(f64::total_cmp);
    let summary = |values: &[f64]| {
        let mid = values.len() / 2;
        json!({"p10": values[3], "median": (values[mid - 1] + values[mid]) / 2.0, "p90": values[28]})
    };
    Ok(json!({
        "load_ms": load_ms,
        "samples": SAMPLES,
        "cache_origin": "synthetic_persistent_state",
        "first_timed_position": START + WARMUP,
        "last_timed_position": START + WARMUP + SAMPLES - 1,
        "visits": PHYSICAL.len(),
        "dispatches_per_step": dispatches,
        "command_buffers_per_step": 1,
        "completion_waits_per_step": 1,
        "logits_bytes_readback": VOCABULARY * 4,
        "wall_ms": summary(&wall),
        "gpu_ms": summary(&gpu_ms),
    }))
}

/// Compare the one-row exact kernels under token-major and layer-major
/// scheduling. Keep state fingerprints and logits private to the comparison.
pub fn measure_prefill(
    checkpoint: &Path,
    oracle: &Path,
    count: usize,
) -> Result<Value> {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let first = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let prefix = first.values::<u32>("start", "U32", &[1])?[0] as usize;
    if !(8..=prefix).contains(&count) {
        return Err("Inkling prefill comparison needs at least eight available tokens");
    }
    let tokens = Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[prefix + 4])?;
    let mut session = Session::load(checkpoint, prefix + 4)?;
    session.capture_draft_features(true);
    let mut lanes = Some(PrefillLanes::new(&mut session.runtime, BATCHED_PREFILL_ROWS)?);
    let mut run = |session: &mut Session, layer_major, offset| -> Result<_> {
        session.reset();
        if layer_major {
            std::mem::swap(&mut session.prefill_lanes, &mut lanes);
        }
        let started = Instant::now();
        session.ingest_many(&tokens[offset..offset + count])?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut state = DefaultHasher::new();
        for visit in &session.visits {
            for cache in [&visit.keys, &visit.values] {
                read_shared::<u16>(cache, count.min(visit.capacity) * KV).hash(&mut state);
            }
            for convolution in &visit.conv {
                read_shared::<u16>(convolution, convolution.length() / 2).hash(&mut state);
            }
        }
        let features = session.draft_features().ok_or("missing captured prefill features")?;
        let snapshot = session.snapshot()?;
        let continuation = session.step(tokens[offset + count])?;
        session.restore(&snapshot)?;
        let restored = session.step(tokens[offset + count])?;
        let replay = continuation.iter().zip(&restored).all(|(a, b)| a.to_bits() == b.to_bits());
        if layer_major {
            std::mem::swap(&mut session.prefill_lanes, &mut lanes);
        }
        Ok((ms, state.finish(), features, continuation, replay))
    };
    let _ = run(&mut session, false, 0)?;
    let _ = run(&mut session, true, 0)?;
    let mut pairs = Vec::new();
    let mut exact_state = true;
    let mut exact_taps = true;
    let mut exact_logits = true;
    let mut exact_replay = true;
    for pair in 0..8 {
        let offset = pair * (prefix - count) / 8;
        let first = run(&mut session, pair % 2 == 1, offset)?;
        let second = run(&mut session, pair % 2 == 0, offset)?;
        let (base, candidate) = if pair % 2 == 0 {
            (first, second)
        } else {
            (second, first)
        };
        exact_state &= base.1 == candidate.1;
        exact_taps &= base.2 == candidate.2;
        exact_logits &= base.3.iter().zip(&candidate.3).all(|(base, candidate)| base.to_bits() == candidate.to_bits());
        exact_replay &= base.4 && candidate.4;
        pairs.push(json!({"base_ms": base.0, "candidate_ms": candidate.0, "ratio": candidate.0 / base.0}));
    }
    let continuation_boundary =
        LOCAL.min((prefix.saturating_sub(BATCHED_PREFILL_ROWS) / BATCHED_PREFILL_ROWS) * BATCHED_PREFILL_ROWS);
    if continuation_boundary == 0 {
        return Err("Inkling prefill comparison needs two complete candidate tiles");
    }
    session.reset();
    session.prefill_lanes = lanes.take();
    session.ingest_many(&tokens[..continuation_boundary])?;
    let snapshot = session.snapshot()?;
    let tail_end = continuation_boundary + BATCHED_PREFILL_ROWS;
    let tail = &tokens[continuation_boundary..tail_end];
    session.ingest_many(tail)?;
    let before_restore = session.step(tokens[tail_end])?;
    session.restore(&snapshot)?;
    session.ingest_many(tail)?;
    let after_restore = session.step(tokens[tail_end])?;
    let exact_tile_replay = before_restore.iter().zip(&after_restore).all(|(a, b)| a.to_bits() == b.to_bits());
    Ok(json!({
        "input_tokens": count,
        "baseline_command_batch_rows": PREFILL_COMMAND_ROWS,
        "candidate_command_batch_rows": BATCHED_PREFILL_ROWS,
        "exact_target_state": exact_state,
        "exact_target_taps": exact_taps,
        "exact_continuation_logits": exact_logits,
        "exact_snapshot_replay": exact_replay,
        "snapshot_tile_boundary": continuation_boundary,
        "exact_snapshot_tile_replay": exact_tile_replay,
        "pairs": pairs,
    }))
}
