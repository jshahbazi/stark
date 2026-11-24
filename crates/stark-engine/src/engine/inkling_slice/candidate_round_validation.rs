// SPDX-License-Identifier: Apache-2.0
//! Replay draft candidate rows through the production source verifier.

use std::{path::Path, time::Instant};

use serde_json::{Value, json};

use super::{
    transaction_validation::{greedy, state_digest},
    *,
};

const CHECKPOINT_STRIDE: usize = 16;

fn read_tokens(
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

#[derive(Default)]
struct Counters {
    rounds: usize,
    emitted_tokens: usize,
    accepted_proposals: usize,
    proposals_checked: usize,
    accepted_by_position: [usize; 7],
    rejected_by_position: [usize; 7],
    transaction_calls: usize,
    singleton_calls: usize,
    rollback_calls: usize,
    rollback_rows: usize,
    target_rows_checked: usize,
    exact_logit_rows: usize,
    top1_drift_rows: usize,
    prefix_state_checks: usize,
    prefix_state_matches: usize,
    committed_state_checks: usize,
    committed_state_matches: usize,
    verify_ms: f64,
    commit_ms: f64,
}

impl Counters {
    fn report(
        &self,
        teacher_len: usize,
    ) -> Value {
        json!({
            "rounds": self.rounds,
            "emitted_tokens": self.emitted_tokens,
            "accepted_proposals": self.accepted_proposals,
            "proposals_checked": self.proposals_checked,
            "accepted_by_position": self.accepted_by_position,
            "rejected_by_position": self.rejected_by_position,
            "transaction_calls": self.transaction_calls,
            "singleton_calls": self.singleton_calls,
            "rollback_calls": self.rollback_calls,
            "rollback_rows": self.rollback_rows,
            "target_rows_checked": self.target_rows_checked,
            "exact_logit_rows": self.exact_logit_rows,
            "top1_drift_rows": self.top1_drift_rows,
            "prefix_state_checks": self.prefix_state_checks,
            "prefix_state_matches": self.prefix_state_matches,
            "committed_state_checks": self.committed_state_checks,
            "committed_state_matches": self.committed_state_matches,
            "verify_ms": self.verify_ms,
            "commit_ms": self.commit_ms,
            "verify_plus_commit_ms": self.verify_ms + self.commit_ms,
            "rounds_plus_accepted": self.rounds + self.accepted_proposals,
            "token_count_invariant": self.rounds + self.accepted_proposals == teacher_len,
        })
    }
}

struct Segment {
    consumed_rows: usize,
    rejected: bool,
}

fn run_segment(
    session: &mut Session,
    inputs: &[u32],
    candidate_row: &[u32],
    candidate_offset: usize,
    checked_proposals: usize,
    teacher_input_start: usize,
    reference_hashes: &[String],
    reference_top1: &[u32],
    counters: &mut Counters,
) -> Result<Segment> {
    let mut mismatch_row = None;
    if inputs.len() == 1 {
        let started = Instant::now();
        let logits = session.step(inputs[0])?;
        counters.verify_ms += started.elapsed().as_secs_f64() * 1000.0;
        counters.singleton_calls += 1;
        let index = teacher_input_start;
        counters.target_rows_checked += 1;
        counters.exact_logit_rows += usize::from(target_validation::native_digest(&logits) == reference_hashes[index]);
        counters.top1_drift_rows += usize::from(greedy(&logits) != reference_top1[index]);
    } else {
        let started = Instant::now();
        let transaction = session.begin_verify(inputs)?;
        counters.verify_ms += started.elapsed().as_secs_f64() * 1000.0;
        counters.transaction_calls += 1;
        for row in 0..inputs.len() {
            let logits = transaction.logits(row)?;
            let index = teacher_input_start + row;
            counters.target_rows_checked += 1;
            counters.exact_logit_rows +=
                usize::from(target_validation::native_digest(logits) == reference_hashes[index]);
            let top1 = greedy(logits);
            counters.top1_drift_rows += usize::from(top1 != reference_top1[index]);
            if row < checked_proposals {
                let proposal_index = candidate_offset + row;
                counters.proposals_checked += 1;
                if top1 == candidate_row[proposal_index] {
                    counters.accepted_by_position[proposal_index] += 1;
                    counters.accepted_proposals += 1;
                } else {
                    counters.rejected_by_position[proposal_index] += 1;
                    mismatch_row = Some(row);
                    break;
                }
            }
        }
        let keep = mismatch_row.map_or(inputs.len(), |row| row + 1);
        counters.rollback_calls += usize::from(keep < inputs.len());
        counters.rollback_rows += inputs.len() - keep;
        let started = Instant::now();
        let _session = transaction.commit(keep)?;
        counters.commit_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(Segment {
        consumed_rows: mismatch_row.map_or(inputs.len(), |row| row + 1),
        rejected: mismatch_row.is_some(),
    })
}

fn sparse_state_check(
    session: &Session,
    boundary: usize,
    reference: &[Option<(String, Vec<u8>)>],
    counters: &mut Counters,
    prefix: bool,
) -> Result<bool> {
    let Some(expected) = reference.get(boundary).and_then(Option::as_ref) else {
        return Ok(true);
    };
    let actual = state_digest(session)?;
    let exact = actual.0 == expected.0 && actual.1 == expected.1;
    if prefix {
        counters.prefix_state_checks += 1;
        counters.prefix_state_matches += usize::from(exact);
    } else {
        counters.committed_state_checks += 1;
        counters.committed_state_matches += usize::from(exact);
    }
    Ok(exact)
}

fn walk(
    session: &mut Session,
    teacher: &[u32],
    proposals: &[u32],
    depth: usize,
    reference_hashes: &[String],
    reference_top1: &[u32],
    reference_states: &[Option<(String, Vec<u8>)>],
) -> Result<Value> {
    let mut counters = Counters::default();
    let mut cursor = 0;
    let mut complete = true;
    while cursor < teacher.len() {
        if !sparse_state_check(session, cursor, reference_states, &mut counters, true)? {
            complete = false;
            break;
        }
        let effective_depth = depth.min(teacher.len() - cursor - 1);
        counters.rounds += 1;
        if effective_depth == 0 {
            run_segment(
                session,
                &[teacher[cursor]],
                &[],
                0,
                0,
                cursor,
                reference_hashes,
                reference_top1,
                &mut counters,
            )?;
            counters.emitted_tokens += 1;
            cursor += 1;
            if !sparse_state_check(session, cursor, reference_states, &mut counters, false)? {
                complete = false;
                break;
            }
            continue;
        }

        let anchor_index = cursor;
        let candidate_row = &proposals[anchor_index * 7..anchor_index * 7 + 7];
        let first_rows = if effective_depth <= 3 {
            effective_depth + 1
        } else {
            4
        };
        let mut first_inputs = Vec::with_capacity(first_rows);
        first_inputs.push(teacher[anchor_index]);
        first_inputs.extend_from_slice(&candidate_row[..first_rows - 1]);
        let first_checks = if effective_depth <= 3 {
            effective_depth
        } else {
            4
        };
        let first = run_segment(
            session,
            &first_inputs,
            candidate_row,
            0,
            first_checks,
            anchor_index,
            reference_hashes,
            reference_top1,
            &mut counters,
        )?;
        counters.emitted_tokens += first.consumed_rows;
        cursor += first.consumed_rows;
        if !sparse_state_check(session, cursor, reference_states, &mut counters, false)? {
            complete = false;
            break;
        }
        if first.rejected {
            continue;
        }
        if effective_depth <= 3 {
            continue;
        }

        let remaining_proposals = effective_depth - 3;
        let second_inputs = &candidate_row[3..3 + remaining_proposals];
        let second_checks = remaining_proposals.saturating_sub(1);
        let second = run_segment(
            session,
            second_inputs,
            candidate_row,
            4,
            second_checks,
            cursor,
            reference_hashes,
            reference_top1,
            &mut counters,
        )?;
        counters.emitted_tokens += second.consumed_rows;
        cursor += second.consumed_rows;
        if !sparse_state_check(session, cursor, reference_states, &mut counters, false)? {
            complete = false;
            break;
        }
    }
    let final_state_exact = sparse_state_check(session, cursor, reference_states, &mut counters, false)?;
    Ok(json!({
        "requested_depth": depth,
        "walk": counters.report(teacher.len()),
        "teacher_cursor": cursor,
        "complete": complete && cursor == teacher.len(),
        "final_state_exact": final_state_exact,
    }))
}

/// Replay candidate proposals at depths three and seven over a common serial
/// teacher continuation. Width seven uses two real transactions of at most four
/// rows; every rejection commits only its accepted input prefix.
pub fn measure(
    checkpoint: &Path,
    draft: &Path,
    trace_path: &Path,
    candidate_path: &Path,
    seven_first: bool,
) -> Result<Value> {
    let trace = Tensors::open(trace_path)?;
    let prompt_len = trace.values::<u32>("prompt_len", "U32", &[1])?[0] as usize;
    let teacher_len = trace.values::<u32>("teacher_len", "U32", &[1])?[0] as usize;
    if prompt_len == 0 || teacher_len == 0 {
        return Err("Inkling teacher trace is empty");
    }
    let all_tokens = read_tokens(&trace, prompt_len + teacher_len)?;
    let prompt = &all_tokens[..prompt_len];
    let teacher = &all_tokens[prompt_len..];
    let candidate = Tensors::open(candidate_path)?;
    let proposals = candidate.values::<u32>("proposals", "U32", &[teacher_len, 7])?;

    let mut session = Session::load(checkpoint, 4352)?;
    session.load_dspark(draft)?;
    session.set_batched_prefill_width(true, true, 8192)?;
    session.set_source_verifier(true)?;
    session.set_dspark_enabled(true)?;
    let initial_logits = super::trace_verify_validation::worker_prefill(&mut session, prompt)?;
    let prefill_anchor_matches_teacher = greedy(&initial_logits) == teacher[0];

    let mut reference_hashes = Vec::with_capacity(teacher_len);
    let mut reference_top1 = Vec::with_capacity(teacher_len);
    let mut reference_states = vec![None; teacher_len + 1];
    reference_states[0] = Some(state_digest(&session)?);
    for (index, &token) in teacher.iter().enumerate() {
        let logits = session.step(token)?;
        reference_hashes.push(target_validation::native_digest(&logits));
        reference_top1.push(greedy(&logits));
        let boundary = index + 1;
        if boundary % CHECKPOINT_STRIDE == 0 || boundary + 1 >= teacher_len || boundary == teacher_len {
            reference_states[boundary] = Some(state_digest(&session)?);
        }
    }

    let (first_depth, second_depth) = if seven_first {
        (7, 3)
    } else {
        (3, 7)
    };
    let mut depth_three = None;
    let mut depth_seven = None;
    for depth in [first_depth, second_depth] {
        session.reset();
        super::trace_verify_validation::worker_prefill(&mut session, prompt)?;
        let result =
            walk(&mut session, teacher, &proposals, depth, &reference_hashes, &reference_top1, &reference_states)?;
        if depth == 3 {
            depth_three = Some(result);
        } else {
            depth_seven = Some(result);
        }
    }
    Ok(json!({
        "prompt_len": prompt_len,
        "teacher_len": teacher_len,
        "prefill_mode": "source_text",
        "prefill_anchor_matches_teacher": prefill_anchor_matches_teacher,
        "production_max_rows": MAX_VERIFY_ROWS,
        "depth_three": depth_three.ok_or("missing depth-three walk")?,
        "depth_seven_offline_chunk_emulation": depth_seven.ok_or("missing depth-seven walk")?,
        "depth_order": if seven_first { [7, 3] } else { [3, 7] },
    }))
}
