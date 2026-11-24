// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only DSpark source-ingress and target-verification control.

use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
    time::Instant,
};

use half::bf16;
use serde_json::{Value, json};

use super::*;

/// Capture a private, native teacher trajectory and the exact four target
/// features consumed by Inkling's DSpark/DFlash target adapter. Input carries
/// the rendered prompt, EOS ID, and optionally a historical continuation.
/// The report exposes aggregate counts and digests.
pub fn capture_dspark_trace(
    checkpoint: &Path,
    draft: &Path,
    input: &Path,
    output: &Path,
) -> Result<Value> {
    const TEACHER_CAP: usize = 192;
    let source = Tensors::open(input)?;
    let prompt_len = source.values::<u32>("prompt_len", "U32", &[1])?[0] as usize;
    if !(PackedPrefill::MIN_ROWS + 4..=4096).contains(&prompt_len) || prompt_len + TEACHER_CAP > 4352 {
        return Err("Inkling DFlash trace prompt must be 20 to 4,096 tokens for the 4,352-token diagnostic context");
    }
    let prompt = source.values::<u32>("tokens", "U32", &[prompt_len])?;
    let eos = source.values::<u32>("eos_token_id", "U32", &[1])?[0];
    let reference_teacher = source
        .contains("reference_teacher")
        .then(|| source.values::<u32>("reference_teacher", "U32", &[TEACHER_CAP]))
        .transpose()?;
    if prompt.iter().any(|&token| token as usize >= VOCABULARY) || eos as usize >= VOCABULARY {
        return Err("Inkling DFlash trace token is outside the vocabulary");
    }
    if reference_teacher.as_ref().is_some_and(|tokens| tokens.iter().any(|&token| token as usize >= VOCABULARY)) {
        return Err("Inkling historical teacher token is outside the vocabulary");
    }

    let mut session = Session::load(checkpoint, 4352)?;
    session.load_dspark(draft)?;
    session.set_dspark_enabled(true)?;
    session.capture_draft_features(true);
    session.set_batched_prefill_width(true, true, 8192)?;
    if !session.supports_source_text_prefill(4352) {
        return Err("Inkling DFlash trace requires the production M5 source-prefill path");
    }
    session.set_source_verifier(true)?;
    let draft_int8_head_prepared = session.prepare_dspark_int8_head().is_ok();
    let block = crate::engine::inkling_slice::dspark::DraftBlock::new(&mut session.runtime)?;
    let mut tokens = Vec::with_capacity(prompt_len + TEACHER_CAP);
    let mut features = Vec::<u16>::with_capacity((prompt_len + TEACHER_CAP) * 4 * WIDTH);
    let mut scores: Vec<f32>;
    let mut proposals = Vec::<Vec<u32>>::with_capacity(TEACHER_CAP);

    // Use the worker's source bulk split at its 2,048-row snapshot grid and
    // four-token suffix, including the short rendered prompts.
    let prefill_end = prompt_len.saturating_sub(4);
    let split = prefill_end / Session::SOURCE_PREFILL_ROWS * Session::SOURCE_PREFILL_ROWS;
    let ranges = [0..split, split..prefill_end];
    let chunk_rows = session.prefill_batch_rows();
    for range in ranges {
        for chunk in prompt[range].chunks(chunk_rows) {
            session.ingest_many(chunk)?;
            let captured = capture_appended_features(&session, chunk.len())?;
            features.extend(captured);
            tokens.extend_from_slice(chunk);
        }
    }
    for &token in &prompt[prefill_end..prompt_len - 1] {
        let _ = session.step(token)?;
        features.extend(session.draft_features().ok_or("Inkling DFlash trace feature capture failed")?);
        tokens.push(token);
    }
    let last = prompt[prompt_len - 1];
    scores = session.step(last)?;
    features.extend(session.draft_features().ok_or("Inkling DFlash trace feature capture failed")?);
    tokens.push(last);

    let mut eos_reached = false;
    let mut reference_top1_matches = 0usize;
    let mut reference_top1_tested = 0usize;
    let mut width2_tested = 0usize;
    let mut width2_matches = 0usize;
    let mut width3_tested = 0usize;
    let mut width3_matches = 0usize;
    let share_kv_width2 = session.runtime.source_text_active && session.runtime.verify_fast_three;
    for index in 0..TEACHER_CAP {
        let native_top1 = greedy(&scores);
        let anchor = reference_teacher.as_ref().map_or(native_top1, |teacher| teacher[index]);
        if reference_teacher.is_some() {
            reference_top1_tested += 1;
            reference_top1_matches += usize::from(native_top1 == anchor);
        }
        if anchor == eos {
            // Keep the EOS row and its target features in the fixture, then stop.
            proposals.push(vec![0; 7]);
            let _ = session.step(anchor)?;
            features.extend(session.draft_features().ok_or("Inkling DFlash trace feature capture failed")?);
            tokens.push(anchor);
            eos_reached = true;
            break;
        }
        let draft_state = session.active_dspark().ok_or("Inkling DSpark state unavailable")?;
        let max_row = draft_state.propose_many(
            &session.runtime,
            anchor,
            session.position,
            &session.embedding,
            &session.embed_norm,
            &session.unembedding,
            7,
            &block,
        )?;
        if [0, 31, 95, 159].contains(&index) {
            let depth3 = draft_state.propose_many(
                &session.runtime,
                anchor,
                session.position,
                &session.embedding,
                &session.embed_norm,
                &session.unembedding,
                3,
                &block,
            )?;
            width3_tested += 1;
            width3_matches += usize::from(depth3.as_slice() == &max_row[..3]);
            let depth2 = draft_state.propose_many(
                &session.runtime,
                anchor,
                session.position,
                &session.embedding,
                &session.embed_norm,
                &session.unembedding,
                2,
                &block,
            )?;
            width2_tested += 1;
            width2_matches += usize::from(depth2.as_slice() == &max_row[..2]);
        }
        proposals.push(max_row);
        scores = session.step(anchor)?;
        features.extend(session.draft_features().ok_or("Inkling DFlash trace feature capture failed")?);
        tokens.push(anchor);
    }

    let mut tried = [0usize; 7];
    let mut accepted = [0usize; 7];
    let teacher_len = tokens.len() - prompt_len;
    for (anchor_index, proposed) in proposals.iter().take(teacher_len).enumerate() {
        for offset in 0..7 {
            let Some(&target) = tokens.get(prompt_len + anchor_index + 1 + offset) else {
                break;
            };
            tried[offset] += 1;
            if proposed[offset] != target {
                break;
            }
            accepted[offset] += 1;
        }
    }

    let rounds = dspark_depth_rounds(&tokens[prompt_len..], &proposals);
    let proposal_tensor = proposals.iter().take(teacher_len).flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
    let feature_hash = super::target_validation::native_digest(&features);
    let token_hash = super::target_validation::native_digest(&tokens);
    let reference_hash = reference_teacher.as_ref().map(|teacher| super::target_validation::native_digest(teacher));
    write_trace_safetensors(output, prompt_len, teacher_len, eos_reached, &tokens, &proposal_tensor, &features)?;
    Ok(json!({
        "prompt_tokens": prompt_len,
        "teacher_tokens": teacher_len,
        "prompt_target_path": "source_worker_schedule",
        "teacher_stop": if eos_reached { "eos" } else { "fixed_cap" },
        "teacher_source": if reference_teacher.is_some() { "historical_worker" } else { "native_greedy" },
        "historical_teacher_sha256": reference_hash,
        "native_top1_matches": reference_top1_matches,
        "native_top1_tested": reference_top1_tested,
        "proposal_anchors": proposals.len(),
        "proposal_depths": [1, 2, 3, 4, 5, 6, 7],
        "conditional_exposures": tried,
        "accepted_by_position": accepted,
        "rounds_for_depths": rounds,
        "width_prefix_controls": {
            "depth2": {"tested": width2_tested, "matched": width2_matches, "share_kv": share_kv_width2},
            "depth3": {"tested": width3_tested, "matched": width3_matches},
        },
        "draft_int8_head_prepared": draft_int8_head_prepared,
        "teacher_tokens_sha256": token_hash,
        "feature_tensor_sha256": feature_hash,
        "fixture": output.file_name().and_then(|name| name.to_str()).unwrap_or("private"),
    }))
}

fn write_trace_safetensors(
    output: &Path,
    prompt_len: usize,
    teacher_len: usize,
    eos_reached: bool,
    tokens: &[u32],
    proposals: &[u32],
    features: &[u16],
) -> Result<()> {
    let sequence_len = tokens.len();
    if features.len() != sequence_len * 4 * WIDTH {
        return Err("Inkling DFlash trace feature geometry is invalid");
    }
    if proposals.len() != teacher_len * 7 {
        return Err("Inkling DFlash proposal geometry is invalid");
    }
    let Some(parent) = output.parent() else {
        return Err("Inkling DFlash trace output directory is invalid");
    };
    fs::create_dir_all(parent).map_err(|_| "cannot create private DFlash trace directory")?;

    let token_bytes = std::mem::size_of_val(tokens);
    let proposal_bytes = std::mem::size_of_val(proposals);
    let feature_bytes = std::mem::size_of_val(features);
    let scalar_bytes = 4;
    let metadata_bytes = 3 * scalar_bytes;
    let header_value = json!({
        "tokens": {"dtype":"U32", "shape":[sequence_len], "data_offsets":[0,token_bytes]},
        "prompt_len": {"dtype":"U32", "shape":[1], "data_offsets":[token_bytes,token_bytes+scalar_bytes]},
        "teacher_len": {"dtype":"U32", "shape":[1], "data_offsets":[token_bytes+scalar_bytes,token_bytes+2*scalar_bytes]},
        "eos_reached": {"dtype":"U32", "shape":[1], "data_offsets":[token_bytes+2*scalar_bytes,token_bytes+3*scalar_bytes]},
        "proposals": {"dtype":"U32", "shape":[teacher_len,7], "data_offsets":[token_bytes+metadata_bytes,token_bytes+metadata_bytes+proposal_bytes]},
        "features": {"dtype":"BF16", "shape":[sequence_len,4,WIDTH], "data_offsets":[token_bytes+metadata_bytes+proposal_bytes,token_bytes+metadata_bytes+proposal_bytes+feature_bytes]}
    });
    let mut header = serde_json::to_vec(&header_value).map_err(|_| "cannot encode DFlash trace header")?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(output)
        .map_err(|_| "cannot create private DFlash trace fixture")?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&(header.len() as u64).to_le_bytes()).map_err(|_| "cannot write DFlash trace header")?;
    writer.write_all(&header).map_err(|_| "cannot write DFlash trace header")?;
    writer.write_all(bytemuck::cast_slice(tokens)).map_err(|_| "cannot write DFlash trace tokens")?;
    writer.write_all(&(prompt_len as u32).to_le_bytes()).map_err(|_| "cannot write DFlash trace metadata")?;
    writer.write_all(&(teacher_len as u32).to_le_bytes()).map_err(|_| "cannot write DFlash trace metadata")?;
    writer.write_all(&(eos_reached as u32).to_le_bytes()).map_err(|_| "cannot write DFlash trace metadata")?;
    writer.write_all(bytemuck::cast_slice(proposals)).map_err(|_| "cannot write DFlash trace proposals")?;
    writer.write_all(bytemuck::cast_slice(features)).map_err(|_| "cannot write DFlash trace features")?;
    writer.flush().map_err(|_| "cannot finish DFlash trace fixture")?;
    writer.get_ref().sync_all().map_err(|_| "cannot sync DFlash trace fixture")?;
    Ok(())
}

fn capture_appended_features(
    session: &Session,
    rows: usize,
) -> Result<Vec<u16>> {
    if !session.runtime.source_text_active || rows < PackedPrefill::MIN_ROWS {
        return Err("Inkling DFlash trace requires a packed source capture chunk");
    }
    session
        .packed_prefill
        .as_ref()
        .and_then(|packed| packed.packed_draft_features(rows))
        .ok_or("Inkling packed DFlash trace features unavailable")
}

fn dspark_depth_rounds(
    teacher: &[u32],
    proposals: &[Vec<u32>],
) -> Value {
    let mut by_depth = Vec::with_capacity(7);
    for depth in 1..=7 {
        let mut index = 0usize;
        let mut round_count = 0usize;
        let mut accepted_count = 0usize;
        let mut emitted_count = 0usize;
        while index < teacher.len() {
            let Some(row) = proposals.get(index) else {
                break;
            };
            let available = (teacher.len() - index - 1).min(depth);
            let mut accepted = 0usize;
            while accepted < available && row[accepted] == teacher[index + accepted + 1] {
                accepted += 1;
            }
            // The current anchor is part of the emitted sequence. This also
            // accounts for the terminal singleton when no proposal is active.
            let emitted = 1 + accepted;
            round_count += 1;
            accepted_count += accepted;
            emitted_count += emitted;
            index += emitted;
        }
        by_depth.push(json!({
            "depth": depth,
            "rounds": round_count,
            "accepted_proposals": accepted_count,
            "emitted_tokens": emitted_count,
            "teacher_tokens_accounted": emitted_count == teacher.len(),
        }));
    }
    Value::Array(by_depth)
}

fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0;
    for token in 1..logits.len() {
        if logits[token] > logits[best] {
            best = token;
        }
    }
    best as u32
}

pub fn measure_dspark(
    checkpoint: &Path,
    draft: &Path,
    fixture: &Path,
    prefix: usize,
    steps: usize,
) -> Result<Value> {
    if prefix == 0 || steps == 0 {
        return Err("Inkling draft validation needs prefix and decode rows");
    }
    let context = prefix.checked_add(steps).and_then(|rows| rows.checked_add(2)).ok_or("Inkling context overflow")?;
    let source = Tensors::open(fixture)?;
    let tokens = source.values::<u32>("tokens", "U32", &[prefix])?;
    let anchors = source.values::<u32>("anchors", "U32", &[4])?;
    let expected = source.values::<u32>("proposals", "U32", &[4])?;
    let taps = source.values::<u16>("taps", "BF16", &[1, prefix, 4 * WIDTH])?;
    if anchors.iter().any(|&token| token as usize >= VOCABULARY) {
        return Err("Inkling draft fixture contains an out-of-range anchor");
    }
    let mut session = Session::load(checkpoint, context)?;
    let load_started = Instant::now();
    session.load_dspark(draft)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let (reference_ingress, source_stages) = {
        let draft = session.dspark.as_ref().ok_or("Inkling draft weights unavailable")?;
        let staging = zeroed(&session.runtime.device, 4 * WIDTH * 2)?;
        for (position, row) in taps.as_chunks::<{ 4 * WIDTH }>().0.iter().enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), staging.contents().as_ptr().cast(), row.len()) };
            execute(&session.runtime, |gpu| draft.commit(gpu, &staging, position))?;
        }
        let mut matches = 0;
        let stage_names = ["draft_hidden", "draft_latent", "draft_feature", "folded_checks", "folded_bias_checks"];
        let mut stage_worst = [0.0f64; 5];
        let stage_values = if stage_names.iter().all(|name| source.contains(name)) {
            Some([
                source.values::<u16>(stage_names[0], "BF16", &[4, WIDTH])?,
                source.values::<u16>(stage_names[1], "BF16", &[4, 256])?,
                source.values::<u16>(stage_names[2], "BF16", &[WIDTH])?,
                source.values::<u16>(stage_names[3], "BF16", &[4, 256])?,
                source.values::<u16>(stage_names[4], "BF16", &[4])?,
            ])
        } else {
            None
        };
        for (index, (&anchor, &want)) in anchors.iter().zip(&expected).enumerate() {
            let got = draft.propose(
                &session.runtime,
                anchor,
                prefix,
                &session.embedding,
                &session.embed_norm,
                &session.unembedding,
            )?;
            matches += usize::from(got == want);
            if let Some(values) = &stage_values {
                for (stage, got) in draft.probe(&session.runtime, anchor as usize)?.into_iter().enumerate() {
                    let start = if stage == 2 {
                        0
                    } else {
                        index * got.len()
                    };
                    let expected = &values[stage][start..start + got.len()];
                    let (noise, signal) = got.into_iter().zip(expected).fold((0.0, 0.0), |(n, s), (a, &b)| {
                        let a = f64::from(bf16::from_bits(a).to_f32());
                        let b = f64::from(bf16::from_bits(b).to_f32());
                        (n + (a - b).powi(2), s + b.powi(2))
                    });
                    stage_worst[stage] = stage_worst[stage].max((noise / signal.max(f64::MIN_POSITIVE)).sqrt());
                }
            }
        }
        (
            matches,
            stage_values.map(|_| {
                stage_names
                    .into_iter()
                    .zip(stage_worst)
                    .map(|(key, value)| (key.to_owned(), json!(value)))
                    .collect::<serde_json::Map<_, _>>()
            }),
        )
    };

    let prefill_started = Instant::now();
    let mut logits = session.prefill(&tokens)?;
    let serial_prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let mut serial = Vec::with_capacity(steps);
    let mut serial_logits = Vec::with_capacity(steps);
    let started = Instant::now();
    for _ in 0..steps {
        let token = greedy(&logits);
        serial.push(token);
        logits = session.step(token)?;
        serial_logits.push(logits.clone());
    }
    let serial_seconds = started.elapsed().as_secs_f64();

    session.reset();
    session.set_dspark_enabled(true)?;
    let prefill_started = Instant::now();
    let mut logits = session.prefill(&tokens)?;
    let draft_prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let mut anchor = greedy(&logits);
    let snapshot = session.snapshot()?;
    let replay_proposal = session.draft_propose(anchor)?;
    let replay_logits = session.step(anchor)?;
    session.reset();
    session.restore(&snapshot)?;
    let restored_proposal = session.draft_propose(anchor)?;
    let restored_logits = session.step(anchor)?;
    let draft_restore_exact = replay_proposal == restored_proposal && replay_logits == restored_logits;
    session.reset();
    session.restore(&snapshot)?;

    let mut target_logit_matches = 0;
    let mut target_token_matches = 0;
    let mut accepted = 0;
    let mut proposals = 0;
    let mut proposal_seconds = 0.0;
    let started = Instant::now();
    for index in 0..steps {
        let next = Instant::now();
        let proposal = if index + 1 < steps {
            session.draft_propose(anchor)?
        } else {
            None
        };
        proposal_seconds += next.elapsed().as_secs_f64();
        target_token_matches += usize::from(anchor == serial[index]);
        logits = session.step(anchor)?;
        target_logit_matches += usize::from(logits == serial_logits[index]);
        anchor = greedy(&logits);
        if let Some(candidate) = proposal {
            proposals += 1;
            accepted += usize::from(candidate == anchor);
        }
    }
    let draft_seconds = started.elapsed().as_secs_f64();

    session.reset();
    session.set_dspark_enabled(false)?;
    let rejects_other_mode = session.restore(&snapshot).is_err();
    let serial_recovery = session.prefill(&tokens).map(|logits| greedy(&logits) == serial[0])?;
    Ok(json!({
        "prefix": prefix,
        "verified_steps": steps,
        "original_draft_load_seconds": load_seconds,
        "source_ingress_proposal_matches": reference_ingress,
        "source_ingress_proposals": expected.len(),
        "source_ingress_stage_relative_l2": source_stages,
        "target_logit_matches": target_logit_matches,
        "target_token_matches": target_token_matches,
        "draft_restore_exact": draft_restore_exact,
        "rejects_other_mode_snapshot": rejects_other_mode,
        "serial_recovery": serial_recovery,
        "proposed": proposals,
        "accepted": accepted,
        "serial_prefill_seconds": serial_prefill_seconds,
        "draft_prefill_seconds": draft_prefill_seconds,
        "serial_decode_seconds": serial_seconds,
        "draft_decode_seconds": draft_seconds,
        "draft_proposal_seconds": proposal_seconds,
    }))
}

pub fn measure_dspark_depth(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prefix: usize,
) -> Result<Value> {
    measure_dspark_depth_mode(checkpoint, draft, oracle, prefix, false)
}

pub fn measure_source_dspark_depth(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prefix: usize,
) -> Result<Value> {
    measure_dspark_depth_mode(checkpoint, draft, oracle, prefix, true)
}

fn measure_dspark_depth_mode(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prefix: usize,
    source_target: bool,
) -> Result<Value> {
    const ROUNDS: usize = 32;
    const DEPTHS: [usize; 3] = [1, 3, 7];
    let extra = if prefix == 4 {
        4
    } else if prefix == 4096 {
        3
    } else {
        return Err("Inkling draft depth diagnostic needs the short or stock 4K fixture");
    };
    let source = Tensors::open(&oracle.join("tokens.safetensors"))?;
    let prompt = source.values::<u32>("tokens", "U32", &[prefix + extra])?;

    let mut session = Session::load(checkpoint, 4352)?;
    session.load_dspark(draft)?;
    let block = crate::engine::inkling_slice::dspark::DraftBlock::new(&mut session.runtime)?;
    session.set_batched_prefill(true, true)?;
    session.set_dspark_enabled(true)?;
    if source_target {
        let packed = session.packed_prefill.as_mut().ok_or("Inkling source prefill unavailable")?;
        if !packed.supports_source_context(4352) {
            return Err("Inkling source DSpark probe requires packed M5 prefill");
        }
        packed.enable_draft_taps(&mut session.runtime)?;
        packed.set_source_scope(true);
        // Only this diagnostic pairs DSpark with the source target precision.
        session.runtime.source_text_active = true;
    }
    let before = Instant::now();
    let mut packed_tap_checks = 0;
    let mut scores = if source_target && prefix == 4096 {
        session.capture_draft_features(true);
        for chunk in [&prompt[..2048], &prompt[2048..4092]] {
            session.ingest_many(chunk)?;
            let recorded = session.draft_features().ok_or("Inkling source DSpark target taps unavailable")?;
            if !session
                .packed_prefill
                .as_ref()
                .is_some_and(|packed| packed.packed_draft_row_matches(chunk.len() - 1, &recorded))
            {
                return Err("Inkling packed DSpark features differ from the final row capture");
            }
            packed_tap_checks += 1;
        }
        session.capture_draft_features(false);
        session.ingest_many(&prompt[4092..4095])?;
        session.step(prompt[4095])?
    } else if source_target {
        session.ingest_many(&prompt[..prefix - 1])?;
        session.step(prompt[prefix - 1])?
    } else {
        session.prefill(&prompt[..prefix])?
    };
    let prefill_ms = before.elapsed().as_secs_f64() * 1000.0;
    let snapshot = session.snapshot()?;
    let mut pinned_unequal = Vec::new();
    if source_target {
        let check = |file: &str, observed: &[f32]| -> Result<usize> {
            let expected = Tensors::open(&oracle.join(file))?;
            let logits = expected.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
            Ok(logits.iter().zip(observed).filter(|(a, b)| a.to_bits() != b.to_bits()).count())
        };
        if prefix == 4096 {
            pinned_unequal.push(check("row-0.safetensors", &scores)?);
        }
        let observed = session.step(prompt[prefix])?;
        pinned_unequal.push(check(
            if prefix == 4096 {
                "row-1.safetensors"
            } else {
                "row-0.safetensors"
            },
            &observed,
        )?);
        session.restore(&snapshot)?;
        if pinned_unequal.iter().any(|&count| count != 0) {
            return Err("Inkling source DSpark target trace differed from pinned logits");
        }
    }
    let mut teacher = Vec::new();
    for _ in 0..ROUNDS + DEPTHS[2] {
        let token = greedy(&scores);
        teacher.push(token);
        scores = session.step(token)?;
    }
    session.restore(&snapshot)?;

    let mut timings = std::array::from_fn::<_, { DEPTHS.len() }, _>(|_| Vec::new());
    let mut tried = [0; DEPTHS[2]];
    let mut accepted = [0; DEPTHS[2]];
    let mut first_mask_exact = 0;
    let mut nested_exact = 0;
    let mut decode_ms = Vec::new();
    for index in 0..ROUNDS {
        let anchor = teacher[index];
        let Some(draft) = session.active_dspark() else {
            return Err("Inkling draft unavailable");
        };
        let mut proposals = std::array::from_fn::<_, { DEPTHS.len() }, _>(|_| Vec::new());
        for offset in 0..DEPTHS.len() {
            let level = (index + offset) % DEPTHS.len();
            let start = Instant::now();
            proposals[level] = draft.propose_many(
                &session.runtime,
                anchor,
                session.position,
                &session.embedding,
                &session.embed_norm,
                &session.unembedding,
                DEPTHS[level],
                &block,
            )?;
            timings[level].push(start.elapsed().as_secs_f64() * 1000.0);
        }
        first_mask_exact += usize::from(session.draft_propose(anchor)? == Some(proposals[0][0]));
        nested_exact +=
            usize::from(DEPTHS.iter().zip(&proposals).all(|(&depth, tokens)| proposals[2][..depth] == *tokens));
        for (offset, (&proposal, &target)) in proposals[2].iter().zip(&teacher[index + 1..]).enumerate() {
            tried[offset] += 1;
            if proposal != target {
                break;
            }
            accepted[offset] += 1;
        }
        let start = Instant::now();
        if greedy(&session.step(anchor)?) != teacher[index + 1] {
            return Err("Inkling draft depth changed serial replay");
        }
        decode_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(
        json!({"prefix":prefix, "source_target":source_target, "pinned_unequal_logits":pinned_unequal, "packed_final_tap_checks":packed_tap_checks, "rounds":ROUNDS, "depths":DEPTHS, "prefill_ms":prefill_ms,
              "proposal_ms":timings, "target_step_ms":decode_ms, "conditional_exposures":tried,
              "accepted_by_position":accepted, "first_mask_matches":first_mask_exact,
              "nested_proposal_matches":nested_exact}),
    )
}
