// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only controls for the transactional source verifier.

use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

fn state_digest(session: &Session) -> Result<(String, Vec<u8>)> {
    let draft = session.active_dspark().ok_or("Inkling transaction proof requires DSpark")?;
    let buffers = draft.state_buffers(session.position);
    let bytes = draft.state_buffers(session.position).map(|(_, length)| length).sum();
    let readback = zeroed(&session.runtime.device, bytes)?;
    let command = session.runtime.queue.command_buffer().ok_or("Inkling draft proof command allocation failed")?;
    let copy = command.blit_command_encoder().ok_or("Inkling draft proof encoder allocation failed")?;
    let mut offset = 0;
    for (buffer, length) in buffers {
        copy.copy_buffer_to_buffer(buffer, 0, &readback, offset, length);
        offset += length;
    }
    copy.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err("Inkling draft proof readback failed");
    }
    let committed = unsafe { std::slice::from_raw_parts(readback.contents().as_ptr().cast::<u8>(), bytes) };
    Ok((target_validation::state_digest(session), Sha256::digest(committed).to_vec()))
}

fn same_logits(
    left: &[f32],
    right: &[f32],
) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn greedy(scores: &[f32]) -> u32 {
    scores
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &score)| {
            if score > best.1 {
                (i as u32, score)
            } else {
                best
            }
        })
        .0
}

fn check_rejections(
    session: &mut Session,
    base: &Snapshot,
    inputs: &[u32],
) -> Result<usize> {
    session.restore(base)?;
    let mut baseline = Vec::with_capacity(MAX_VERIFY_ROWS + 1);
    for &token in &inputs[..=MAX_VERIFY_ROWS] {
        let logits = session.step(token)?;
        baseline.push((logits, state_digest(session)?));
    }
    let mut controls = 0;
    for width in 2..=MAX_VERIFY_ROWS {
        for kept in 1..=width {
            session.restore(base)?;
            let mut tokens = inputs[..width].to_vec();
            for token in &mut tokens[kept..] {
                *token = (*token + 17) % VOCABULARY as u32;
            }
            let txn = session.begin_verify(&tokens)?;
            for (row, (logits, _)) in baseline.iter().take(kept).enumerate() {
                if !same_logits(txn.logits(row)?, logits) {
                    return Err("Inkling rollback proof changed a verified row");
                }
            }
            txn.commit(kept)?;
            if state_digest(session)? != baseline[kept - 1].1 {
                return Err("Inkling rollback proof changed committed target or draft state");
            }
            let next = session.step(inputs[kept])?;
            if !same_logits(&next, &baseline[kept].0) || state_digest(session)? != baseline[kept].1 {
                return Err("Inkling rollback proof changed the next continuation");
            }
            controls += 1;
        }
    }
    Ok(controls)
}

fn screen_verifier(
    session: &mut Session,
    base: &Snapshot,
    teacher: &[u32],
) -> Result<Vec<Value>> {
    if !session.verifier.as_ref().is_some_and(|workspace| workspace.shares_local_attention()) {
        return Ok(Vec::new());
    }
    let options = [0, 2, 4];
    session.restore(base)?;
    let expected = teacher[..MAX_VERIFY_ROWS].iter().map(|&id| session.step(id)).collect::<Result<Vec<_>>>()?;
    let mut report = Vec::new();
    for width in 2..=MAX_VERIFY_ROWS {
        let mut observations = vec![Vec::new(); options.len()];
        for trial in 0..8 {
            for offset in 0..options.len() {
                let arm = (trial + offset) % options.len();
                session.restore(base)?;
                let local_rows = options[arm];
                session.verifier.as_mut().ok_or("missing verifier workspace")?.local_rows_per_group = local_rows;
                let start = Instant::now();
                let txn = session.begin_verify(&teacher[..width])?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if trial == 0 {
                    for (row, want) in expected.iter().take(width).enumerate() {
                        if !same_logits(txn.logits(row)?, want) {
                            return Err("Inkling verifier candidate changed timed row logits");
                        }
                    }
                }
                txn.commit(1)?;
                observations[arm].push(ms);
            }
        }
        for (local_rows, wall_ms) in options.iter().zip(observations) {
            report.push(json!({"width":width,"local_rows_per_group":local_rows,"wall_ms":wall_ms}));
        }
    }
    session.verifier.as_mut().ok_or("missing verifier workspace")?.local_rows_per_group = 2;
    Ok(report)
}

fn screen_run_width(
    session: &mut Session,
    base: &Snapshot,
    teacher: &[u32],
) -> Result<Vec<Value>> {
    const CONTROL: Kernel = "ink_verify_nvfp4_group_dot";
    let candidate = session.runtime.verify_nvfp4_kernel;
    if candidate == CONTROL || session.runtime.verify_half_act {
        return Ok(Vec::new());
    }
    session.restore(base)?;
    let expected = teacher[..MAX_VERIFY_ROWS].iter().map(|&id| session.step(id)).collect::<Result<Vec<_>>>()?;
    let mut report = Vec::new();
    for width in 2..=MAX_VERIFY_ROWS {
        let mut timings = [Vec::new(), Vec::new()];
        for trial in 0..8 {
            for arm in [trial % 2, 1 - trial % 2] {
                session.restore(base)?;
                session.runtime.verify_nvfp4_kernel = [CONTROL, candidate][arm];
                let start = Instant::now();
                let txn = session.begin_verify(&teacher[..width])?;
                timings[arm].push(start.elapsed().as_secs_f64() * 1000.0);
                for (row, want) in expected.iter().take(width).enumerate() {
                    if !same_logits(txn.logits(row)?, want) {
                        return Err("Inkling fixed run width changed verifier logits");
                    }
                }
                txn.commit(1)?;
            }
        }
        report.push(json!({"width":width,"ordinary_ms":timings[0],"fixed_width_ms":timings[1]}));
    }
    session.runtime.verify_nvfp4_kernel = candidate;
    Ok(report)
}

fn screen_draft_shortlist(
    session: &mut Session,
    base: &Snapshot,
    teacher: &[u32],
) -> Result<Vec<Value>> {
    if !session.runtime.draft_shortlist {
        return Ok(Vec::new());
    }
    session.restore(base)?;
    let mut timings = [const { [const { Vec::new() }; 2] }; MAX_VERIFY_ROWS - 1];
    let mut changed = [0; MAX_VERIFY_ROWS - 1];
    let mut accepted = [[0; MAX_VERIFY_ROWS - 1]; 2];
    for index in 0..64 {
        let anchor = teacher[index];
        for depth in 1..MAX_VERIFY_ROWS {
            let mut proposals = [Vec::new(), Vec::new()];
            for arm in [index % 2, 1 - index % 2] {
                session.runtime.draft_shortlist = arm == 1;
                let start = Instant::now();
                proposals[arm] = session.propose_verify(anchor, depth)?;
                timings[depth - 1][arm].push(start.elapsed().as_secs_f64() * 1000.0);
                if depth == MAX_VERIFY_ROWS - 1 {
                    let matches = proposals[arm]
                        .iter()
                        .zip(&teacher[index + 1..])
                        .take_while(|(proposal, target)| proposal == target)
                        .count();
                    for count in accepted[arm].iter_mut().take(matches) {
                        *count += 1;
                    }
                }
            }
            changed[depth - 1] += usize::from(proposals[0] != proposals[1]);
        }
        session.ingest_many(&[anchor])?;
    }
    session.runtime.draft_shortlist = true;
    Ok(timings
        .into_iter()
        .enumerate()
        .map(|(i, [ordinary, shortlist])| {
            json!({
                "depth":i+1, "ordinary_ms":ordinary, "shortlist_ms":shortlist,
                "different_proposal_rounds":changed[i],
                "depth_three_teacher_matches":(i==MAX_VERIFY_ROWS-2).then_some(accepted),
            })
        })
        .collect())
}

fn screen_compact_runs(
    session: &mut Session,
    base: &Snapshot,
    teacher: &[u32],
) -> Result<Vec<Value>> {
    let experimental = session.runtime.verify_compact_runs;
    if experimental == 0 {
        return Ok(Vec::new());
    }
    session.restore(base)?;
    let expected = teacher[..MAX_VERIFY_ROWS].iter().map(|&id| session.step(id)).collect::<Result<Vec<_>>>()?;
    let mut report = Vec::new();
    for width in 2..=MAX_VERIFY_ROWS {
        let mut timings = [const { Vec::new() }; 3];
        for trial in 0..8 {
            for offset in 0..timings.len() {
                let arm = (trial + offset) % timings.len();
                session.restore(base)?;
                session.runtime.verify_compact_runs = arm;
                let start = Instant::now();
                let txn = session.begin_verify(&teacher[..width])?;
                timings[arm].push(start.elapsed().as_secs_f64() * 1000.0);
                for (row, want) in expected.iter().take(width).enumerate() {
                    if !same_logits(txn.logits(row)?, want) {
                        return Err("Inkling compact verifier runs changed F32 logits");
                    }
                }
                txn.commit(1)?;
            }
        }
        for (arm, wall_ms) in timings.into_iter().enumerate() {
            report.push(json!({"width":width,"arm":arm,"wall_ms":wall_ms}));
        }
    }
    session.runtime.verify_compact_runs = experimental;
    Ok(report)
}

fn screen_fused_finish(
    session: &mut Session,
    base: &Snapshot,
    teacher: &[u32],
    setting: fn(&mut Runtime) -> &mut bool,
) -> Result<Vec<Value>> {
    if !*setting(&mut session.runtime) {
        return Ok(Vec::new());
    }
    session.restore(base)?;
    let expected = teacher[..MAX_VERIFY_ROWS].iter().map(|&id| session.step(id)).collect::<Result<Vec<_>>>()?;
    let mut report = Vec::new();
    for width in 2..=MAX_VERIFY_ROWS {
        let mut timings = [Vec::new(), Vec::new()];
        for trial in 0..8 {
            for arm in [trial % 2, 1 - trial % 2] {
                session.restore(base)?;
                *setting(&mut session.runtime) = arm == 1;
                let start = Instant::now();
                let txn = session.begin_verify(&teacher[..width])?;
                timings[arm].push(start.elapsed().as_secs_f64() * 1000.0);
                for (row, want) in expected.iter().take(width).enumerate() {
                    if !same_logits(txn.logits(row)?, want) {
                        return Err("Inkling fused verifier finish changed F32 logits");
                    }
                }
                txn.commit(1)?;
            }
        }
        report.push(json!({"width":width,"ordinary_ms":timings[0],"fused_ms":timings[1]}));
    }
    *setting(&mut session.runtime) = true;
    Ok(report)
}

pub fn measure_verify_transaction(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prefix: usize,
) -> Result<Value> {
    const OUTPUTS: usize = 96;
    let extra = match prefix {
        4 => 4,
        4096 => 3,
        _ => return Err("Inkling transaction proof requires a short or stock 4K capture"),
    };
    let fixture = Tensors::open(&oracle.join("tokens.safetensors"))?;
    let prompt = fixture.values::<u32>("tokens", "U32", &[prefix + extra])?;
    let mut session = Session::load(checkpoint, 4352)?;
    session.load_dspark(draft)?;
    session.set_batched_prefill(true, true)?;
    let before_verifier = session.runtime.device.current_allocated_size();
    session.set_source_verifier(true)?;
    let verifier_bytes = session.runtime.device.current_allocated_size().saturating_sub(before_verifier);
    session.set_dspark_enabled(true)?;
    let cap_crossing = if prefix == 4096 {
        session.ingest_many(&prompt[..2048])?;
        session.ingest_many(&prompt[2048..4092])?;
        session.ingest_many(&prompt[4092..4095])?;
        Some(session.snapshot()?)
    } else {
        session.ingest_many(&prompt[..prefix - 1])?;
        None
    };
    let mut logits = session.step(prompt[prefix - 1])?;
    if prefix == 4096 {
        let fixture_seed = Tensors::open(&oracle.join("row-0.safetensors"))?;
        if !same_logits(&logits, &fixture_seed.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?) {
            return Err("Inkling verifier prefill seed differed from pinned logits");
        }
    }
    let base = session.snapshot()?;
    let first = greedy(&logits);
    let mut teacher = vec![first];
    let started = Instant::now();
    while teacher.len() < OUTPUTS {
        logits = session.step(*teacher.last().unwrap())?;
        teacher.push(greedy(&logits));
    }
    let serial_ms = started.elapsed().as_secs_f64() * 1000.0;
    let serial_state = state_digest(&session)?;
    let controls = check_rejections(&mut session, &base, &teacher)?;
    let crossing_controls = if let Some(snapshot) = cap_crossing {
        let inputs = std::iter::once(prompt[4095]).chain(teacher.iter().copied()).collect::<Vec<_>>();
        check_rejections(&mut session, &snapshot, &inputs)?
    } else {
        0
    };

    let candidate_commands = screen_verifier(&mut session, &base, &teacher)?;
    let fixed_run_commands = screen_run_width(&mut session, &base, &teacher)?;
    let shortlist_drafts = screen_draft_shortlist(&mut session, &base, &teacher)?;
    let compact_runs = screen_compact_runs(&mut session, &base, &teacher)?;
    let fused_finish = screen_fused_finish(&mut session, &base, &teacher, |runtime| &mut runtime.verify_fused_finish)?;
    let fused_attention =
        screen_fused_finish(&mut session, &base, &teacher, |runtime| &mut runtime.verify_fused_attention)?;
    let mut rounds = Vec::new();
    for depth in 1..MAX_VERIFY_ROWS {
        session.restore(&base)?;
        let mut output = vec![first];
        let mut proposed = 0;
        let mut accepted = 0;
        let mut rounds_at_depth = [0; MAX_VERIFY_ROWS];
        let mut draft_ms = 0.0;
        let mut verify_ms = 0.0;
        let mut finish_ms = 0.0;
        let started = Instant::now();
        while output.len() < OUTPUTS {
            let anchor = *output.last().unwrap();
            let next_depth = depth.min(OUTPUTS - output.len() - 1);
            rounds_at_depth[next_depth] += 1;
            if next_depth == 0 {
                output.push(greedy(&session.step(anchor)?));
                continue;
            }
            let start = Instant::now();
            let proposals = session.propose_verify(anchor, next_depth)?;
            draft_ms += start.elapsed().as_secs_f64() * 1000.0;
            proposed += proposals.len();
            let inputs = std::iter::once(anchor).chain(proposals.iter().copied()).collect::<Vec<_>>();
            let start = Instant::now();
            let txn = session.begin_verify(&inputs)?;
            verify_ms += start.elapsed().as_secs_f64() * 1000.0;
            let mut kept = 1;
            for row in 0..inputs.len() {
                let token = greedy(txn.logits(row)?);
                output.push(token);
                if proposals.get(row) != Some(&token) {
                    break;
                }
                kept += 1;
            }
            let start = Instant::now();
            txn.commit(kept)?;
            finish_ms += start.elapsed().as_secs_f64() * 1000.0;
            accepted += kept - 1;
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if output != teacher || state_digest(&session)? != serial_state {
            return Err("Inkling speculative round changed serial output or committed state");
        }
        rounds.push(json!({
            "depth": depth, "round_counts": rounds_at_depth, "proposed": proposed, "accepted": accepted,
            "draft_ms": draft_ms, "verify_ms": verify_ms, "finish_ms": finish_ms,
            "elapsed_ms": elapsed_ms, "decode_tok_s": 1000.0 * (OUTPUTS - 1) as f64 / elapsed_ms,
            "serial_match": true,
        }));
    }
    // A caller that abandons a provisional transaction cannot use its state.
    session.restore(&base)?;
    drop(session.begin_verify(&teacher[..2])?);
    if session.position != 0 {
        return Err("Inkling abandoned transaction retained a live position");
    }
    session.restore(&base)?;
    session.reset();
    session.set_source_verifier(false)?;
    if session.restore(&base).is_ok() {
        return Err("Inkling source draft snapshot was accepted in legacy mode");
    }

    Ok(json!({
        "prefix": prefix, "output_tokens": OUTPUTS, "additional_metal_bytes": verifier_bytes, "rejection_controls": controls,
        "crossing_controls": crossing_controls, "abandoned_transaction_reset": true, "cross_mode_snapshot_rejected": true,
        "serial_ms": serial_ms, "serial_decode_tok_s": 1000.0 * (OUTPUTS - 1) as f64 / serial_ms,
        "rounds": rounds, "candidate_commands": candidate_commands, "fixed_run_commands": fixed_run_commands, "shortlist_drafts": shortlist_drafts, "compact_runs": compact_runs, "fused_finish": fused_finish, "fused_attention": fused_attention,
    }))
}
