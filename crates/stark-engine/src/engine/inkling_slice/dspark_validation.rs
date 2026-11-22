// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only DSpark source-ingress and target-verification control.

use std::time::Instant;

use half::bf16;
use serde_json::{Value, json};

use super::*;

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
        for (position, row) in taps.chunks_exact(4 * WIDTH).enumerate() {
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
