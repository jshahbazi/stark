// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only timing of the production Inkling source-verifier transaction.

use std::{path::Path, time::Instant};

use serde_json::{Value, json};

use super::{
    transaction_validation::{greedy, same_logits, state_digest},
    *,
};

const WIDTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const REPEATS: usize = 4;
const MAX_POSITIONS: usize = 6;
const SOURCE_SUFFIX: usize = 4;

pub(super) fn worker_prefill(
    session: &mut Session,
    prompt: &[u32],
) -> Result<Vec<f32>> {
    if prompt.is_empty() {
        return Err("Inkling worker prefill needs a nonempty prompt");
    }
    session.set_text_prefill_scope(true);
    let last = prompt.len() - 1;
    let bulk_end = prompt.len().saturating_sub(SOURCE_SUFFIX);
    let boundary = bulk_end / Session::SOURCE_PREFILL_ROWS * Session::SOURCE_PREFILL_ROWS;
    if boundary > 0 {
        session.ingest_many(&prompt[..boundary])?;
    }
    if bulk_end > boundary {
        session.ingest_many(&prompt[boundary..bulk_end])?;
    }
    if last > bulk_end {
        session.ingest_many(&prompt[bulk_end..last])?;
    }
    session.step(prompt[last])
}

fn tokens(
    trace: &Tensors,
    count: usize,
) -> Result<Vec<u32>> {
    trace.values::<u32>("tokens", "U32", &[count]).or_else(|_| {
        trace
            .values::<i32>("tokens", "I32", &[count])?
            .into_iter()
            .map(|id| u32::try_from(id).map_err(|_| "negative Inkling token ID"))
            .collect()
    })
}

/// Measure actual source-verifier transactions using externally supplied
/// teacher rows. Widths above four are explicitly offline chunk emulations:
/// the production worker currently admits at most four rows per transaction.
pub fn measure(
    checkpoint: &Path,
    draft: &Path,
    trace_path: &Path,
) -> Result<Value> {
    let trace = Tensors::open(trace_path)?;
    let prompt_len = trace.values::<u32>("prompt_len", "U32", &[1])?[0] as usize;
    let teacher_len = trace.values::<u32>("teacher_len", "U32", &[1])?[0] as usize;
    if prompt_len == 0 || teacher_len < 8 || !trace.contains("features") {
        return Err("Inkling teacher trace lacks the required prompt, rows or private features");
    }
    let all_tokens = tokens(&trace, prompt_len + teacher_len)?;
    let teacher = &all_tokens[prompt_len..];
    let mut session = Session::load(checkpoint, 4352)?;
    session.load_dspark(draft)?;
    session.set_batched_prefill_width(true, true, 8192)?;
    session.set_source_verifier(true)?;
    session.set_dspark_enabled(true)?;
    let prefill_logits = worker_prefill(&mut session, &all_tokens[..prompt_len])?;
    let prefill_anchor_matches_teacher = greedy(&prefill_logits) == teacher[0];

    let anchor_count = teacher.len() - 8 + 1;
    let position_count = anchor_count.min(MAX_POSITIONS);
    let stride = anchor_count.div_ceil(position_count).max(1);
    let mut positions = Vec::with_capacity(position_count);

    for anchor_index in (0..anchor_count).step_by(stride).take(position_count) {
        let base = session.snapshot()?;
        let rows = &teacher[anchor_index..anchor_index + 8];
        let mut serial_logits = Vec::with_capacity(8);
        let mut serial_state = Vec::with_capacity(8);
        for &token in rows {
            serial_logits.push(session.step(token)?);
            serial_state.push(state_digest(&session)?);
        }
        session.restore(&base)?;

        let mut measurements: [Vec<Value>; 8] = std::array::from_fn(|_| Vec::new());
        let mut exact_logits = [false; 8];
        let mut exact_target_state = [false; 8];
        let mut exact_draft_state = [false; 8];
        let mut top1_drift_rows = [0usize; 8];

        for trial in 0..=REPEATS {
            // Trial zero warms every width. Multiplication by five rotates the
            // width order across both trials and trace positions.
            for offset in 0..WIDTHS.len() {
                let width_index = (offset + trial * 3 + anchor_index * 5) % WIDTHS.len();
                let width = WIDTHS[width_index];
                session.restore(&base)?;
                let mut consumed = 0;
                let mut verify_ms = 0.0;
                let mut commit_ms = 0.0;
                let mut observed = (trial == 0).then(|| Vec::with_capacity(width));

                while consumed < width {
                    let chunk = (width - consumed).min(MAX_VERIFY_ROWS);
                    if chunk == 1 {
                        let started = Instant::now();
                        let logits = session.step(rows[consumed])?;
                        verify_ms += started.elapsed().as_secs_f64() * 1000.0;
                        if let Some(observed) = observed.as_mut() {
                            observed.push(logits);
                        }
                        consumed += 1;
                        continue;
                    }

                    let started = Instant::now();
                    let transaction = session.begin_verify(&rows[consumed..consumed + chunk])?;
                    verify_ms += started.elapsed().as_secs_f64() * 1000.0;
                    if let Some(observed) = observed.as_mut() {
                        for row in 0..chunk {
                            observed.push(transaction.logits(row)?.to_vec());
                        }
                    }
                    let started = Instant::now();
                    let _session = transaction.commit(chunk)?;
                    commit_ms += started.elapsed().as_secs_f64() * 1000.0;
                    consumed += chunk;
                }

                if trial == 0 {
                    let observed = observed.as_ref().ok_or("Inkling verifier warmup did not retain checks")?;
                    exact_logits[width_index] = observed
                        .iter()
                        .zip(&serial_logits[..width])
                        .all(|(actual, expected)| same_logits(actual, expected));
                    top1_drift_rows[width_index] = observed
                        .iter()
                        .zip(&serial_logits[..width])
                        .filter(|(actual, expected)| greedy(actual) != greedy(expected))
                        .count();
                    let (target, draft_state) = state_digest(&session)?;
                    exact_target_state[width_index] = target == serial_state[width - 1].0;
                    exact_draft_state[width_index] = draft_state == serial_state[width - 1].1;
                } else {
                    measurements[width_index].push(json!({
                        "verify_ms": verify_ms,
                        "commit_ms": commit_ms,
                        "total_ms": verify_ms + commit_ms,
                    }));
                }
            }
        }

        session.restore(&base)?;
        session.step(rows[0])?;
        positions.push(json!({
            "teacher_position": anchor_index,
            "widths": WIDTHS.iter().enumerate().map(|(index, &width)| json!({
                "rows": width,
                "support": if width <= MAX_VERIFY_ROWS { "production" } else { "offline_chunk_emulation" },
                "chunks": (0..width).step_by(MAX_VERIFY_ROWS).map(|start| (width - start).min(MAX_VERIFY_ROWS)).collect::<Vec<_>>(),
                "measurements": measurements[index],
                "exact_logits": exact_logits[index],
                "exact_target_state": exact_target_state[index],
                "exact_draft_state": exact_draft_state[index],
                "top1_drift_rows": top1_drift_rows[index],
            })).collect::<Vec<_>>(),
        }));
    }

    Ok(json!({
        "prompt_len": prompt_len,
        "teacher_len": teacher_len,
        "prefill_anchor_matches_teacher": prefill_anchor_matches_teacher,
        "prefill_mode": "source_text",
        "production_max_rows": MAX_VERIFY_ROWS,
        "positions": positions,
    }))
}
