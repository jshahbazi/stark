// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only checks at a caller-chosen prompt length and context.
//!
//! These diagnostics reproduce the generation worker's cold text schedule for
//! an `N`-token prompt. With the source path active, bulk commands of at most
//! `prefill_batch_rows()` rows cover the first `N - 4` tokens, one command
//! covers the three-row tail and the seed at position `N - 1` is a singleton
//! step. Otherwise the bulk covers the first `N - 1` tokens before the seed.
//!
//! Private captures follow the pinned-runtime convention: `row-0.safetensors`
//! records the seed at `start = N - 1`, rows 1 to 3 record the following
//! teacher rows and `tokens.safetensors` holds `N + 3` token IDs. A directory
//! holding only `tokens.safetensors` supports the checks that compare Stark
//! with itself. Reports contain aggregates and SHA-256 fingerprints only.
//!
//! The existing fixed-shape diagnostics are unchanged; these functions are
//! reachable only through their own example subcommands.

use std::{ops::Range, time::Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;

/// Rows held back from the source bulk: the three-row tail and the seed.
const SOURCE_SUFFIX: usize = 4;
/// Teacher rows after the seed in the capture convention.
const TEACHER_ROWS: usize = 3;
/// Fixed prefix-reuse boundaries replayed whenever a bulk command ends there.
/// Global attention log scaling first applies at position 8,192.
const PREFIX_BOUNDARIES: [usize; 2] = [4096, 8192];
const VERIFY_OUTPUTS: usize = 96;
/// The worker.s source floor: every non-empty text prompt.
const SOURCE_PREFILL_MIN_INPUT: usize = 1;

/// Mirror of the generation worker's text-length admission for source
/// prefill (`stark-generate`, `InklingBackend::source_text_prefill`). The
/// worker additionally requires a supported context, which the session
/// applies through its own readiness check. This copy lets the target-only
/// checks select the same precision scope as the worker; it is not the
/// worker's code and does not qualify the worker.
pub(super) fn worker_admits_source(prompt: usize) -> bool {
    prompt >= SOURCE_PREFILL_MIN_INPUT
}

/// Command ranges before the seed row for a `prompt`-token request, in the
/// order the generation worker submits them on a cold request with a prefix
/// cache scope. On the source path the worker ends one segment at its
/// 2,048-aligned snapshot boundary (`Session::SOURCE_PREFILL_ROWS`) and
/// splits each segment into commands of at most `rows`.
pub(super) fn worker_ranges(
    prompt: usize,
    source: bool,
    rows: usize,
) -> Vec<Range<usize>> {
    assert!(rows > 0);
    let Some(last) = prompt.checked_sub(1) else {
        return Vec::new();
    };
    let bulk_end = if source {
        prompt.saturating_sub(SOURCE_SUFFIX)
    } else {
        last
    };
    let boundary = if source {
        bulk_end / Session::SOURCE_PREFILL_ROWS * Session::SOURCE_PREFILL_ROWS
    } else {
        0
    };
    let segment = |from: usize, to: usize| (from..to).step_by(rows).map(move |start| start..to.min(start + rows));
    let mut ranges = segment(0, boundary).chain(segment(boundary, bulk_end)).collect::<Vec<_>>();
    if bulk_end < last {
        ranges.push(bulk_end..last);
    }
    ranges
}

/// Command ends whose snapshots the checks replay: the fixed 4,096 and 8,192
/// boundaries and, on the source path, the worker's own prefix-reuse boundary.
pub(super) fn replay_boundaries(
    ranges: &[Range<usize>],
    prompt: usize,
    source: bool,
    rows: usize,
) -> Vec<usize> {
    let _ = rows;
    let worker = if source {
        prompt.saturating_sub(SOURCE_SUFFIX) / Session::SOURCE_PREFILL_ROWS * Session::SOURCE_PREFILL_ROWS
    } else {
        0
    };
    ranges
        .iter()
        .map(|range| range.end)
        .filter(|&end| end > 0 && (PREFIX_BOUNDARIES.contains(&end) || end == worker))
        .collect()
}

struct Prefilled {
    source: bool,
    rows: usize,
    ranges: Vec<Range<usize>>,
    boundaries: Vec<(usize, Snapshot)>,
    ms: f64,
}

impl Prefilled {
    fn commands(&self) -> Vec<usize> {
        self.ranges.iter().map(ExactSizeIterator::len).collect()
    }
}

fn ingest_worker_schedule(
    session: &mut Session,
    tokens: &[u32],
    prompt: usize,
) -> Result<Prefilled> {
    let source = session.runtime.source_text_active;
    let rows = session.prefill_batch_rows();
    let ranges = worker_ranges(prompt, source, rows);
    let wanted = replay_boundaries(&ranges, prompt, source, rows);
    let mut boundaries = Vec::with_capacity(wanted.len());
    let mut ms = 0.0;
    for range in &ranges {
        let started = Instant::now();
        session.ingest_many(&tokens[range.clone()])?;
        ms += 1000.0 * started.elapsed().as_secs_f64();
        if wanted.contains(&range.end) {
            boundaries.push((range.end, session.snapshot()?));
        }
    }
    Ok(Prefilled {
        source,
        rows,
        ranges,
        boundaries,
        ms,
    })
}

/// Restore each boundary snapshot, submit the remaining commands of the same
/// schedule and compare the seed logits and the committed state.
fn replay_prefix_boundaries(
    session: &mut Session,
    tokens: &[u32],
    prompt: usize,
    prefilled: &Prefilled,
    seed_logits: &[f32],
    seed_state: &str,
    state: fn(&Session) -> Result<String>,
) -> Result<(bool, Vec<Value>)> {
    let mut all = true;
    let mut reports = Vec::with_capacity(prefilled.boundaries.len());
    for (boundary, snapshot) in &prefilled.boundaries {
        session.restore(snapshot)?;
        let restored_position = session.position == *boundary;
        for range in prefilled.ranges.iter().filter(|range| range.start >= *boundary) {
            session.ingest_many(&tokens[range.clone()])?;
        }
        let logits = session.step(tokens[prompt - 1])?;
        let exact_logits = transaction_validation::same_logits(&logits, seed_logits);
        let exact_state = state(session)? == seed_state;
        all &= restored_position && exact_logits && exact_state;
        reports.push(json!({
            "boundary": boundary, "restored_position": restored_position,
            "exact_seed_logits": exact_logits, "exact_state": exact_state,
        }));
    }
    Ok((all, reports))
}

fn read_tokens(
    oracle: &Path,
    prompt: usize,
) -> Result<Vec<u32>> {
    let length = prompt.checked_add(TEACHER_ROWS).ok_or("Inkling prompt length overflow")?;
    Tensors::open(&oracle.join("tokens.safetensors"))?.values::<u32>("tokens", "U32", &[length])
}

/// Whether the directory holds a pinned capture whose seed is this prompt's
/// last token. A capture for another prompt length is an error.
fn has_capture(
    oracle: &Path,
    prompt: usize,
) -> Result<bool> {
    let path = oracle.join("row-0.safetensors");
    if !path.exists() {
        return Ok(false);
    }
    let start = Tensors::open(&path)?.values::<u32>("start", "U32", &[1])?[0] as usize;
    if start + 1 != prompt {
        return Err("Inkling capture seed position differs from the prompt length");
    }
    Ok(true)
}

fn pinned_logits(
    oracle: &Path,
    row: usize,
) -> Result<Vec<f32>> {
    Tensors::open(&oracle.join(format!("row-{row}.safetensors")))?.values::<f32>(
        "full.logits",
        "F32",
        &[1, 1, VOCABULARY],
    )
}

fn unequal(
    got: &[f32],
    want: &[f32],
) -> usize {
    got.iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + got.len().abs_diff(want.len())
}

fn top(values: &[f32]) -> Option<usize> {
    values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index)
}

fn target_state(session: &Session) -> Result<String> {
    Ok(target_validation::state_digest(session))
}

/// Target caches together with the committed draft state.
fn joint_state(session: &Session) -> Result<String> {
    let (target, draft) = transaction_validation::state_digest(session)?;
    let mut digest = Sha256::new();
    digest.update(target.as_bytes());
    digest.update(&draft);
    Ok(format!("{:x}", digest.finalize()))
}

/// Compare the full target with a private pinned capture under the worker's
/// cold schedule at any prompt length and context.
pub fn measure_full_schedule(
    checkpoint: &Path,
    oracle: &Path,
    context: usize,
    exact_prefill: bool,
) -> Result<Value> {
    let first = Tensors::open(&oracle.join("row-0.safetensors"))?;
    let seed = first.values::<u32>("start", "U32", &[1])?[0] as usize;
    let prompt = seed + 1;
    if seed < SOURCE_SUFFIX || prompt.checked_add(TEACHER_ROWS).is_none_or(|length| length > context) {
        return Err("full oracle prompt is outside the context");
    }
    let tokens = read_tokens(oracle, prompt)?;
    let started = Instant::now();
    let mut session = Session::load(checkpoint, context)?;
    session.runtime.exact_prefill = exact_prefill;
    session.set_batched_prefill(true, true)?;
    let admitted = worker_admits_source(prompt);
    session.set_text_prefill_scope(admitted);
    let load_ms = 1000.0 * started.elapsed().as_secs_f64();
    let workspace_bytes = session.source_context_workspace_bytes(context);
    let prefilled = ingest_worker_schedule(&mut session, &tokens, prompt)?;
    let prefill_state = target_validation::state_digest(&session);
    session.capture_draft_features(true);
    session.capture_visits = true;
    let snapshot = session.snapshot()?;
    let mut rows = Vec::with_capacity(TEACHER_ROWS + 1);
    let mut seed_logits = None;
    let mut seed_state = String::new();
    let mut all_logits_exact = true;
    let mut all_top_tokens_match = true;
    for (row, &token) in tokens[seed..].iter().enumerate() {
        let source = Tensors::open(&oracle.join(format!("row-{row}.safetensors")))?;
        let history_5 = target_validation::probe_history(&session, &source, 5)?;
        let history_11 = target_validation::probe_history(&session, &source, 11)?;
        let started = Instant::now();
        let got = session.step(token)?;
        let elapsed_ms = 1000.0 * started.elapsed().as_secs_f64();
        if row == 0 {
            seed_state = target_validation::state_digest(&session);
        }
        let want = source.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?;
        let got_hidden = target_validation::as_float(&read_shared::<u16>(&session.hidden, WIDTH));
        let want_hidden = target_validation::oracle_bf16(&source, "full.hidden", &[1, 1, WIDTH])?;
        let top_token_matches = top(&got) == top(&want);
        all_logits_exact &= unequal(&got, &want) == 0;
        all_top_tokens_match &= top_token_matches;
        rows.push(json!({
            "position": seed + row,
            "milliseconds": elapsed_ms,
            "top_token_matches": top_token_matches,
            "native_logits_sha256": target_validation::native_digest(&got),
            "native_taps_sha256": target_validation::native_digest(
                &session.draft_features().ok_or("missing full target taps")?,
            ),
            "logits": target_validation::report(&got, &want),
            "hidden": target_validation::report(&got_hidden, &want_hidden),
            "partial_state": target_validation::probe_partials(&session, &source)?,
            "history_physical_5": history_5,
            "history_physical_11": history_11,
            "first_pass_physical_4": target_validation::probe_first_pass(&session, &source, 4)?,
            "first_pass_physical_5": target_validation::probe_first_pass(&session, &source, 5)?,
            "first_pass_physical_10": target_validation::probe_first_pass(&session, &source, 10)?,
            "first_pass_physical_11": target_validation::probe_first_pass(&session, &source, 11)?,
            "isolated_epilogue": target_validation::probe_epilogue(&session, &source)?,
        }));
        if row == 0 {
            seed_logits = Some(got);
        }
    }
    let seed_logits = seed_logits.ok_or("empty oracle")?;
    session.restore(&snapshot)?;
    let exact_restore = transaction_validation::same_logits(&session.step(tokens[seed])?, &seed_logits);
    let (exact_boundaries, boundary_replays) =
        replay_prefix_boundaries(&mut session, &tokens, prompt, &prefilled, &seed_logits, &seed_state, target_state)?;
    session.capture_draft_features(false);
    Ok(json!({
        "prompt_tokens": prompt,
        "exact_prefill_reference": exact_prefill,
        "context": context,
        "seed_position": seed,
        "worker_source_admitted": admitted,
        "source_text_active": prefilled.source,
        "prefill_command_rows": prefilled.rows,
        "prefill_commands": prefilled.commands(),
        "context_workspace_additional_bytes": workspace_bytes,
        "load_ms": load_ms,
        "prefill_ms": prefilled.ms,
        "prefill_native_cache_sha256": prefill_state,
        "all_logits_exact": all_logits_exact,
        "all_top_tokens_match": all_top_tokens_match,
        "exact_snapshot_replay": exact_restore,
        "exact_boundary_replays": exact_boundaries,
        "boundary_replays": boundary_replays,
        "rows": rows,
    }))
}

/// Run the worker schedule against Stark alone and record fingerprints for
/// paired comparisons, plus end-of-prefill and prefix-boundary replay.
pub fn measure_schedule_replay(
    checkpoint: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    exact_prefill: bool,
    command_rows: usize,
) -> Result<Value> {
    if prompt <= SOURCE_SUFFIX || prompt.checked_add(TEACHER_ROWS).is_none_or(|length| length > context) {
        return Err("Inkling schedule replay prompt is outside the context");
    }
    let tokens = read_tokens(oracle, prompt)?;
    let mut session = Session::load(checkpoint, context)?;
    session.runtime.exact_prefill = exact_prefill;
    session.set_batched_prefill_width(true, true, command_rows)?;
    let admitted = worker_admits_source(prompt);
    session.set_text_prefill_scope(admitted);
    let workspace_bytes = session.source_context_workspace_bytes(context);
    session.capture_draft_features(true);
    let prefilled = ingest_worker_schedule(&mut session, &tokens, prompt)?;
    let prefill_state = target_validation::state_digest(&session);
    let snapshot = session.snapshot()?;
    let mut rows = Vec::with_capacity(TEACHER_ROWS + 1);
    let mut seed_logits = None;
    let mut seed_state = String::new();
    for (row, &token) in tokens[prompt - 1..].iter().enumerate() {
        let got = session.step(token)?;
        let state = target_validation::state_digest(&session);
        rows.push(json!({
            "position": prompt - 1 + row,
            "native_logits_sha256": target_validation::native_digest(&got),
            "native_taps_sha256": target_validation::native_digest(
                &session.draft_features().ok_or("missing Inkling target taps")?,
            ),
            "native_cache_sha256": state,
        }));
        if row == 0 {
            seed_logits = Some(got);
            seed_state = state;
        }
    }
    let seed_logits = seed_logits.ok_or("empty Inkling schedule")?;
    session.restore(&snapshot)?;
    let exact_restore = transaction_validation::same_logits(&session.step(tokens[prompt - 1])?, &seed_logits);
    let (exact_boundaries, boundary_replays) =
        replay_prefix_boundaries(&mut session, &tokens, prompt, &prefilled, &seed_logits, &seed_state, target_state)?;
    Ok(json!({
        "prompt_tokens": prompt,
        "exact_prefill_reference": exact_prefill,
        "context": context,
        "worker_source_admitted": admitted,
        "source_text_active": prefilled.source,
        "prefill_command_rows": prefilled.rows,
        "prefill_commands": prefilled.commands(),
        "context_workspace_additional_bytes": workspace_bytes,
        "prefill_ms": prefilled.ms,
        "prefill_native_cache_sha256": prefill_state,
        "exact_snapshot_replay": exact_restore,
        "exact_boundary_replays": exact_boundaries,
        "boundary_replays": boundary_replays,
        "rows": rows,
    }))
}

/// Transactional source verifier controls at any prompt length and context:
/// every rejection width from the seed and from the pre-seed snapshot,
/// speculative greedy rounds against serial greedy decoding, prefix-boundary
/// replay with the draft enabled, abandoned transactions and mode isolation.
pub fn measure_verify_transaction_schedule(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    exact_prefill: bool,
) -> Result<Value> {
    if prompt <= SOURCE_SUFFIX
        || prompt.checked_add(VERIFY_OUTPUTS + MAX_VERIFY_ROWS).is_none_or(|end| end > context)
        || prompt + TEACHER_ROWS > context
    {
        return Err("Inkling transaction schedule needs room for the prompt and 96 verified outputs");
    }
    let tokens = read_tokens(oracle, prompt)?;
    let pinned_seed = if has_capture(oracle, prompt)? {
        Some(pinned_logits(oracle, 0)?)
    } else {
        None
    };
    let mut session = Session::load(checkpoint, context)?;
    session.runtime.exact_prefill = exact_prefill;
    session.load_dspark(draft)?;
    session.set_batched_prefill(true, true)?;
    let workspace_bytes = session.source_context_workspace_bytes(context);
    let before_verifier = session.runtime.device.current_allocated_size();
    session.set_source_verifier(true)?;
    let verifier_bytes = session.runtime.device.current_allocated_size().saturating_sub(before_verifier);
    session.prepare_dspark_int8_head()?;
    session.set_dspark_enabled(true)?;
    if !session.runtime.source_text_active {
        return Err("Inkling transaction schedule requires the active source path");
    }
    let prefilled = ingest_worker_schedule(&mut session, &tokens, prompt)?;
    let crossing = session.snapshot()?;
    let mut logits = session.step(tokens[prompt - 1])?;
    let seed_matches_pinned = pinned_seed.as_ref().map(|want| transaction_validation::same_logits(&logits, want));
    let seed_logits_sha256 = target_validation::native_digest(&logits);
    let seed_logits = logits.clone();
    let seed_state = joint_state(&session)?;
    let base = session.snapshot()?;
    let (exact_boundaries, boundary_replays) =
        replay_prefix_boundaries(&mut session, &tokens, prompt, &prefilled, &seed_logits, &seed_state, joint_state)?;
    session.restore(&base)?;

    let first = transaction_validation::greedy(&logits);
    let mut teacher = vec![first];
    let started = Instant::now();
    while teacher.len() < VERIFY_OUTPUTS {
        logits = session.step(*teacher.last().unwrap())?;
        teacher.push(transaction_validation::greedy(&logits));
    }
    let serial_ms = 1000.0 * started.elapsed().as_secs_f64();
    let serial_state = transaction_validation::state_digest(&session)?;
    let serial_state_sha256 = joint_state(&session)?;
    let controls = transaction_validation::check_rejections(&mut session, &base, &teacher)?;
    let inputs = std::iter::once(tokens[prompt - 1]).chain(teacher.iter().copied()).collect::<Vec<_>>();
    let crossing_controls = transaction_validation::check_rejections(&mut session, &crossing, &inputs)?;

    let mut rounds = Vec::new();
    for depth in 1..MAX_VERIFY_ROWS {
        session.restore(&base)?;
        let mut output = vec![first];
        let mut proposed = 0;
        let mut accepted = 0;
        let mut rounds_at_depth = [0; MAX_VERIFY_ROWS];
        let started = Instant::now();
        while output.len() < VERIFY_OUTPUTS {
            let anchor = *output.last().unwrap();
            let next_depth = depth.min(VERIFY_OUTPUTS - output.len() - 1);
            rounds_at_depth[next_depth] += 1;
            if next_depth == 0 {
                output.push(transaction_validation::greedy(&session.step(anchor)?));
                continue;
            }
            let (proposals, txn) = if session.overlaps_verify_for(next_depth) {
                let (proposals, txn, _, _) = session.begin_overlapped_verify(anchor, next_depth)?;
                (proposals, txn)
            } else {
                let proposals = session.propose_verify(anchor, next_depth)?;
                let inputs = std::iter::once(anchor).chain(proposals.iter().copied()).collect::<Vec<_>>();
                let txn = session.begin_verify(&inputs)?;
                (proposals, txn)
            };
            proposed += proposals.len();
            let mut kept = 1;
            for row in 0..=proposals.len() {
                let token = transaction_validation::greedy(txn.logits(row)?);
                output.push(token);
                if proposals.get(row) != Some(&token) {
                    break;
                }
                kept += 1;
            }
            txn.commit(kept)?;
            accepted += kept - 1;
        }
        let elapsed_ms = 1000.0 * started.elapsed().as_secs_f64();
        if output != teacher || transaction_validation::state_digest(&session)? != serial_state {
            return Err("Inkling speculative round changed serial output or committed state");
        }
        rounds.push(json!({
            "depth": depth, "round_counts": rounds_at_depth, "proposed": proposed, "accepted": accepted,
            "elapsed_ms": elapsed_ms, "serial_match": true,
        }));
    }
    session.restore(&base)?;
    if session.overlaps_verify() {
        let (_, txn, _, _) = session.begin_overlapped_verify(teacher[0], 2)?;
        drop(txn);
    } else {
        drop(session.begin_verify(&teacher[..2])?);
    }
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
        "prompt_tokens": prompt,
        "exact_prefill_reference": exact_prefill,
        "context": context,
        "seed_position": prompt - 1,
        "first_verified_position": prompt,
        "last_verified_position": prompt + VERIFY_OUTPUTS - 2,
        "worker_source_admitted": worker_admits_source(prompt),
        "source_text_active": prefilled.source,
        "prefill_command_rows": prefilled.rows,
        "prefill_commands": prefilled.commands(),
        "context_workspace_additional_bytes": workspace_bytes,
        "verifier_additional_metal_bytes": verifier_bytes,
        "output_tokens": VERIFY_OUTPUTS,
        "seed_matches_pinned": seed_matches_pinned,
        "seed_logits_sha256": seed_logits_sha256,
        "serial_teacher_sha256": target_validation::native_digest(&teacher),
        "serial_state_sha256": serial_state_sha256,
        "serial_ms": serial_ms,
        "rejection_controls": controls,
        "crossing_controls": crossing_controls,
        "exact_boundary_replays": exact_boundaries,
        "boundary_replays": boundary_replays,
        "rounds": rounds,
        "abandoned_transaction_reset": true,
        "cross_mode_snapshot_rejected": true,
    }))
}

/// Source-precision DSpark depth check at any prompt length and context.
/// Packed draft taps are compared at every bulk command of at least the
/// packed minimum, the seed and first teacher row are compared with the pinned
/// capture when present, and nested proposals must not change serial replay.
pub fn measure_source_dspark_depth_schedule(
    checkpoint: &Path,
    draft: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
    exact_prefill: bool,
) -> Result<Value> {
    const ROUNDS: usize = 32;
    const DEPTHS: [usize; 3] = [1, 3, 7];
    if prompt <= SOURCE_SUFFIX
        || prompt.checked_add(ROUNDS + DEPTHS[2] + 1).is_none_or(|end| end > context)
        || prompt + TEACHER_ROWS > context
    {
        return Err("Inkling draft depth schedule needs room for the prompt and replay rounds");
    }
    let tokens = read_tokens(oracle, prompt)?;
    let captured = has_capture(oracle, prompt)?;
    let mut session = Session::load(checkpoint, context)?;
    session.runtime.exact_prefill = exact_prefill;
    session.load_dspark(draft)?;
    let block = crate::engine::inkling_slice::dspark::DraftBlock::new(&mut session.runtime)?;
    session.set_batched_prefill(true, true)?;
    session.set_dspark_enabled(true)?;
    let workspace_bytes = session.source_context_workspace_bytes(context);
    let packed = session.packed_prefill.as_mut().ok_or("Inkling source prefill unavailable")?;
    // The forced source scope below bypasses the session's readiness check,
    // so require a global workspace that covers this context.
    if !packed.source_ready(context) {
        return Err("Inkling source DSpark schedule requires a source workspace covering the context");
    }
    packed.enable_draft_taps(&mut session.runtime)?;
    packed.set_source_scope(true);
    // Only this diagnostic pairs DSpark with the source target precision.
    session.runtime.source_text_active = true;
    let rows = session.prefill_batch_rows();
    let ranges = worker_ranges(prompt, true, rows);
    let before = Instant::now();
    let mut packed_tap_checks = 0;
    for range in &ranges {
        let chunk = &tokens[range.clone()];
        let check = chunk.len() >= PackedPrefill::MIN_ROWS;
        session.capture_draft_features(check);
        session.ingest_many(chunk)?;
        if check {
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
    }
    session.capture_draft_features(false);
    let mut scores = session.step(tokens[prompt - 1])?;
    let prefill_ms = 1000.0 * before.elapsed().as_secs_f64();
    let snapshot = session.snapshot()?;
    let pinned_unequal = if captured {
        let seed = unequal(&scores, &pinned_logits(oracle, 0)?);
        let next = unequal(&session.step(tokens[prompt])?, &pinned_logits(oracle, 1)?);
        session.restore(&snapshot)?;
        if seed != 0 || next != 0 {
            return Err("Inkling source DSpark target trace differed from pinned logits");
        }
        Some([seed, next])
    } else {
        None
    };
    let mut teacher = Vec::new();
    for _ in 0..ROUNDS + DEPTHS[2] {
        let token = transaction_validation::greedy(&scores);
        teacher.push(token);
        scores = session.step(token)?;
    }
    session.restore(&snapshot)?;

    let mut tried = [0; DEPTHS[2]];
    let mut accepted = [0; DEPTHS[2]];
    let mut first_mask_exact = 0;
    let mut nested_exact = 0;
    for index in 0..ROUNDS {
        let anchor = teacher[index];
        let Some(draft) = session.active_dspark() else {
            return Err("Inkling draft unavailable");
        };
        let mut proposals = std::array::from_fn::<_, { DEPTHS.len() }, _>(|_| Vec::new());
        for offset in 0..DEPTHS.len() {
            let level = (index + offset) % DEPTHS.len();
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
        if transaction_validation::greedy(&session.step(anchor)?) != teacher[index + 1] {
            return Err("Inkling draft depth changed serial replay");
        }
    }
    Ok(json!({
        "prompt_tokens": prompt,
        "exact_prefill_reference": exact_prefill,
        "context": context,
        "seed_position": prompt - 1,
        "worker_source_admitted": worker_admits_source(prompt),
        "prefill_command_rows": rows,
        "prefill_commands": ranges.iter().map(ExactSizeIterator::len).collect::<Vec<_>>(),
        "context_workspace_additional_bytes": workspace_bytes,
        "pinned_unequal_logits": pinned_unequal,
        "packed_final_tap_checks": packed_tap_checks,
        "prefill_ms": prefill_ms,
        "rounds": ROUNDS,
        "depths": DEPTHS,
        "conditional_exposures": tried,
        "accepted_by_position": accepted,
        "first_mask_matches": first_mask_exact,
        "nested_proposal_matches": nested_exact,
    }))
}

/// Former gate of 20261121-INKPREFILL-07, kept in the report for reference:
/// the candidate keeps at least the incumbent's top-token agreement with the
/// pinned capture, its mean relative logit L2 stays within two points and no
/// row loses five.
const TOLERANCE_MEAN_POINTS: f64 = 0.02;
const TOLERANCE_ROW_POINTS: f64 = 0.05;
/// Noise-floor control: visits whose residual receives a one-unit BF16 flip,
/// cycled across trials. Positions and elements are spread deterministically.
const CONTROL_VISITS: [usize; 12] = [1, 4, 8, 12, 17, 22, 27, 32, 37, 41, 45, 47];
/// The default path is clearly worse than the control if its mean row L2
/// exceeds this multiple of the control trials' median.
const CONTROL_MEDIAN_FACTOR: f64 = 2.0;

struct ToleranceArm {
    prefill_ms: f64,
    logits: Vec<Vec<f32>>,
    partials: Vec<Vec<f32>>,
    state: String,
}

fn relative_l2(
    got: &[f32],
    want: &[f32],
) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (&got, &want) in got.iter().zip(want) {
        signal += f64::from(want).powi(2);
        noise += (f64::from(got) - f64::from(want)).powi(2);
    }
    noise.sqrt() / signal.sqrt().max(f64::MIN_POSITIVE)
}

/// Pinned per-visit BF16 residual blocks for one teacher row.
fn pinned_partials(
    source: &Tensors,
    visits: usize,
) -> Result<Vec<f32>> {
    let mut all = Vec::with_capacity(visits * WIDTH);
    for index in 0..visits {
        all.extend(target_validation::oracle_bf16(source, &format!("full.visit.{index:02}.partial"), &[1, 1, WIDTH])?);
    }
    Ok(all)
}

fn residual_differences(
    got: &[f32],
    want: &[f32],
) -> Vec<usize> {
    got.as_chunks::<WIDTH>().0.iter().zip(want.as_chunks::<WIDTH>().0).map(|(a, b)| unequal(a, b)).collect()
}

fn tolerance_arm(
    session: &mut Session,
    tokens: &[u32],
    ranges: &[Range<usize>],
    seed: usize,
    exact: bool,
    perturb: Option<(usize, usize, usize)>,
) -> Result<ToleranceArm> {
    session.runtime.exact_prefill = exact;
    session.runtime.prefill_perturb = perturb;
    session.capture_visits = false;
    session.reset();
    let started = Instant::now();
    let ingested = ranges.iter().try_for_each(|range| session.ingest_many(&tokens[range.clone()]));
    let prefill_ms = 1000.0 * started.elapsed().as_secs_f64();
    session.runtime.exact_prefill = false;
    session.runtime.prefill_perturb = None;
    ingested?;
    let state = target_validation::state_digest(session);
    session.capture_visits = true;
    let mut logits = Vec::with_capacity(TEACHER_ROWS + 1);
    let mut partials = Vec::with_capacity(TEACHER_ROWS + 1);
    for &token in &tokens[seed..seed + TEACHER_ROWS + 1] {
        logits.push(session.step(token)?);
        partials.push(
            session.visits.iter().flat_map(|visit| read_shared::<f32>(&visit.scratch.partial_debug, WIDTH)).collect(),
        );
    }
    session.capture_visits = false;
    Ok(ToleranceArm {
        prefill_ms,
        logits,
        partials,
        state,
    })
}

/// Per-row metrics of one arm against the pinned capture.
struct ArmRows {
    l2: Vec<f64>,
    top1: Vec<bool>,
    residuals: Vec<Vec<usize>>,
}

fn arm_rows(
    arm: &ToleranceArm,
    pinned: &[Vec<f32>],
    pinned_residuals: &[Vec<f32>],
) -> ArmRows {
    ArmRows {
        l2: arm.logits.iter().zip(pinned).map(|(got, want)| relative_l2(got, want)).collect(),
        top1: arm.logits.iter().zip(pinned).map(|(got, want)| top(got) == top(want)).collect(),
        residuals: arm
            .partials
            .iter()
            .zip(pinned_residuals)
            .map(|(got, want)| residual_differences(got, want))
            .collect(),
    }
}

fn arm_json(rows: &ArmRows) -> Value {
    json!({
        "row_relative_logit_l2": rows.l2,
        "row_top1_matches_pinned": rows.top1,
        "row_differing_residuals_per_visit": rows.residuals,
    })
}

/// Aggregate over every teacher row of every input: mean and worst row L2
/// and the fraction of rows whose top token differs from the pinned capture.
fn aggregate(rows: &[&ArmRows]) -> [f64; 3] {
    let l2 = rows.iter().flat_map(|arm| arm.l2.iter().copied()).collect::<Vec<_>>();
    let flips = rows.iter().flat_map(|arm| arm.top1.iter()).filter(|&&matches| !matches).count();
    [
        l2.iter().sum::<f64>() / l2.len().max(1) as f64,
        l2.iter().copied().fold(0.0, f64::max),
        flips as f64 / l2.len().max(1) as f64,
    ]
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// Tolerance report for the default prefill against the pinned capture,
/// against Stark's own bit-exact reference kernels and against a noise-floor
/// control: the exact kernels with one residual element raised by one BF16
/// unit at one visit, over `trials` locations per input. The default path
/// passes if its mean row L2, worst row L2 and top-token flip rate lie inside
/// the range spanned by the control trials and its mean is at most twice the
/// control median. Captures whose seed follows a multiple of 512 rows replay
/// the pinned 512-row bulk geometry; others use the worker's cold schedule.
/// Only aggregate metrics leave the process.
pub fn measure_prefill_tolerance(
    checkpoint: &Path,
    oracles: &[std::path::PathBuf],
    context: usize,
    trials: usize,
) -> Result<Value> {
    if oracles.is_empty() {
        return Err("Inkling tolerance check requires at least one capture");
    }
    let mut session = Session::load(checkpoint, context)?;
    session.set_batched_prefill(true, true)?;
    session.set_text_prefill_scope(true);
    if !session.runtime.source_text_active {
        return Err("Inkling tolerance check requires the active source path");
    }
    let rows = session.prefill_batch_rows();
    session.runtime.router_probe = Some(grouped_pilot::RouterProbe::new(&session.runtime.device, rows)?);
    let (mut all_router_differing, mut all_router_rows) = (0u64, 0u64);
    let mut inputs = Vec::with_capacity(oracles.len());
    let mut default_rows = Vec::with_capacity(oracles.len());
    let mut exact_rows = Vec::with_capacity(oracles.len());
    let mut control_rows = (0..trials).map(|_| Vec::with_capacity(oracles.len())).collect::<Vec<_>>();
    let mut former_gate = true;
    let mut all_mask_free = true;
    for (index, oracle) in oracles.iter().enumerate() {
        let start = Tensors::open(&oracle.join("row-0.safetensors"))?.values::<u32>("start", "U32", &[1])?[0] as usize;
        let prompt = start + 1;
        if start < SOURCE_SUFFIX || prompt + TEACHER_ROWS > context {
            return Err("Inkling tolerance capture is outside the context");
        }
        let tokens = read_tokens(oracle, prompt)?;
        let ranges = if start.is_multiple_of(512) {
            (0..start).step_by(512).map(|first| first..start.min(first + 512)).collect::<Vec<_>>()
        } else {
            worker_ranges(prompt, true, rows)
        };
        // Perturb only rows inside packed source commands.
        let packed_end =
            ranges.iter().filter(|range| range.len() >= PackedPrefill::SOURCE_MIN_ROWS).map(|range| range.end).max();
        let packed_end = packed_end.ok_or("Inkling tolerance capture has no packed source command")?;
        let mut pinned = Vec::with_capacity(TEACHER_ROWS + 1);
        let mut pinned_residuals = Vec::with_capacity(TEACHER_ROWS + 1);
        for row in 0..=TEACHER_ROWS {
            let source = Tensors::open(&oracle.join(format!("row-{row}.safetensors")))?;
            pinned.push(source.values::<f32>("full.logits", "F32", &[1, 1, VOCABULARY])?);
            pinned_residuals.push(pinned_partials(&source, session.visits.len())?);
        }
        // Alternate which arm runs first across inputs.
        let probe_counts =
            |session: &Session| session.runtime.router_probe.as_ref().map_or([0, 0], |p| p.take_counts());
        probe_counts(&session);
        let exact_first = index % 2 == 1;
        let first = tolerance_arm(&mut session, &tokens, &ranges, start, exact_first, None)?;
        let second = tolerance_arm(&mut session, &tokens, &ranges, start, !exact_first, None)?;
        let (exact, fast) = if exact_first {
            (first, second)
        } else {
            (second, first)
        };
        // Router selections of the default arm against the exact source
        // router on the same inputs, over every routed prefill visit.
        let [router_differing, router_rows] = probe_counts(&session);
        all_router_differing += u64::from(router_differing);
        all_router_rows += u64::from(router_rows);
        // The mask-free global attention must equal the explicit mask path.
        session.runtime.global_mask_reference = true;
        let masked = tolerance_arm(&mut session, &tokens, &ranges, start, false, None);
        session.runtime.global_mask_reference = false;
        let masked = masked?;
        let mask_free_global_identical = masked.state == fast.state
            && masked.logits.iter().zip(&fast.logits).all(|(a, b)| transaction_validation::same_logits(a, b));
        all_mask_free &= mask_free_global_identical;
        let fast_metrics = arm_rows(&fast, &pinned, &pinned_residuals);
        let exact_metrics = arm_rows(&exact, &pinned, &pinned_residuals);
        let mut controls = Vec::with_capacity(trials);
        for (trial, slot) in control_rows.iter_mut().enumerate() {
            let visit = CONTROL_VISITS[trial % CONTROL_VISITS.len()];
            let position = packed_end * (2 * trial + 1) / (2 * trials);
            let element = (trial * 911 + 37) % WIDTH;
            let arm = tolerance_arm(&mut session, &tokens, &ranges, start, true, Some((visit, position, element)))?;
            let metrics = arm_rows(&arm, &pinned, &pinned_residuals);
            controls.push(json!({
                "visit": visit, "position": position, "element": element,
                "changed_state": arm.state != exact.state,
                "metrics": arm_json(&metrics),
            }));
            slot.push(metrics);
        }
        let self_rows = (0..=TEACHER_ROWS)
            .map(|row| {
                json!({
                    "relative_logit_l2": relative_l2(&fast.logits[row], &exact.logits[row]),
                    "top1_matches": top(&fast.logits[row]) == top(&exact.logits[row]),
                    "differing_residuals_per_visit": residual_differences(&fast.partials[row], &exact.partials[row]),
                })
            })
            .collect::<Vec<_>>();
        let count = (TEACHER_ROWS + 1) as f64;
        let fast_mean = fast_metrics.l2.iter().sum::<f64>() / count;
        let exact_mean = exact_metrics.l2.iter().sum::<f64>() / count;
        let worst_row =
            fast_metrics.l2.iter().zip(&exact_metrics.l2).map(|(f, e)| f - e).fold(f64::NEG_INFINITY, f64::max);
        let fast_top = fast_metrics.top1.iter().filter(|&&m| m).count();
        let exact_top = exact_metrics.top1.iter().filter(|&&m| m).count();
        let former = fast_top >= exact_top
            && fast_mean - exact_mean <= TOLERANCE_MEAN_POINTS
            && worst_row <= TOLERANCE_ROW_POINTS;
        former_gate &= former;
        inputs.push(json!({
            "seed_position": start,
            "prefill_commands": ranges.iter().map(ExactSizeIterator::len).collect::<Vec<_>>(),
            "exact_first": exact_first,
            "fast_prefill_ms": fast.prefill_ms,
            "exact_prefill_ms": exact.prefill_ms,
            "fast_equals_exact_state": fast.state == exact.state,
            "mask_free_global_identical": mask_free_global_identical,
            "router_rows_compared": router_rows,
            "router_rows_with_different_experts": router_differing,
            "default_vs_pinned": arm_json(&fast_metrics),
            "exact_vs_pinned": arm_json(&exact_metrics),
            "default_vs_exact": self_rows,
            "noise_floor_controls": controls,
            "former_inkprefill07": {
                "fast_top1_matches_pinned": fast_top, "exact_top1_matches_pinned": exact_top,
                "fast_mean_relative_logit_l2": fast_mean, "exact_mean_relative_logit_l2": exact_mean,
                "worst_row_l2_increase": worst_row, "gate_pass": former,
            },
        }));
        default_rows.push(fast_metrics);
        exact_rows.push(exact_metrics);
    }
    let default_aggregate = aggregate(&default_rows.iter().collect::<Vec<_>>());
    let exact_aggregate = aggregate(&exact_rows.iter().collect::<Vec<_>>());
    let control_aggregates =
        control_rows.iter().map(|rows| aggregate(&rows.iter().collect::<Vec<_>>())).collect::<Vec<_>>();
    let column = |i: usize| control_aggregates.iter().map(|values| values[i]).collect::<Vec<_>>();
    let span = |i: usize| {
        let values = column(i);
        (values.iter().copied().fold(f64::INFINITY, f64::min), values.iter().copied().fold(0.0, f64::max))
    };
    let control_median = median(&mut column(0));
    let inside = (0..3).all(|i| default_aggregate[i] <= span(i).1);
    let not_clearly_worse = default_aggregate[0] <= CONTROL_MEDIAN_FACTOR * control_median;
    let noise_floor_pass = trials > 0 && inside && not_clearly_worse;
    Ok(json!({
        "context": context,
        "fast_nvfp4_kernels": session.runtime.fast_nvfp4,
        "control_trials_per_input": trials,
        "aggregate_fields": ["mean_row_relative_logit_l2", "worst_row_relative_logit_l2", "top1_flip_rate"],
        "default_aggregate": default_aggregate,
        "exact_aggregate": exact_aggregate,
        "control_trial_aggregates": control_aggregates,
        "control_spans": [span(0), span(1), span(2)],
        "control_median_mean_row_l2": control_median,
        "default_inside_control_span": inside,
        "default_within_twice_control_median": not_clearly_worse,
        "noise_floor_pass": noise_floor_pass,
        "mask_free_global_identical": all_mask_free,
        "router_rows_compared": all_router_rows,
        "router_rows_with_different_experts": all_router_differing,
        "former_inkprefill07_all_pass": former_gate,
        "former_gate": {"mean_points": TOLERANCE_MEAN_POINTS, "row_points": TOLERANCE_ROW_POINTS},
        "inputs": inputs,
    }))
}

/// Split `rows` into chunks following a repeating width pattern.
fn split_ranges(
    rows: usize,
    pattern: &[usize],
) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for &width in pattern.iter().cycle() {
        if start >= rows {
            break;
        }
        let end = rows.min(start + width);
        let merge = ranges.last().is_some_and(|last| end - start < PackedPrefill::MIN_ROWS && end - last.start <= 2048);
        if !merge {
            ranges.push(start..end);
        } else if let Some(last) = ranges.last_mut() {
            last.end = end;
        }
        start = end;
    }
    ranges
}

/// Row independence: prefill the same rows under several command splits
/// and compare the committed state and the next-token logits, for the
/// default kernels and for the bit-exact reference kernels as a control.
pub fn measure_split_consistency(
    checkpoint: &Path,
    oracle: &Path,
    prompt: usize,
    context: usize,
) -> Result<Value> {
    // The default path computes every stage at one arithmetic for any packed
    // width except the router, which keeps the source split-K reduction for
    // commands of at most 1,024 rows (a fixed split failed the noise-floor
    // gate on the 512-row holdouts). Splits are therefore compared within
    // two width classes; each class crosses the exact reference's other
    // thresholds (65 rows, the 2,048-row projection tiling). The exact
    // reference is reported as a control and is expected to differ.
    const ROWS: usize = 6000;
    const GROUPS: [&[&[usize]]; 2] = [
        &[&[2048], &[1200], &[1100, 1900], &[1777, 1234, 1489, 1500]],
        &[&[1024], &[16, 1000, 64, 65, 500, 355], &[333, 17, 674]],
    ];
    let tokens = read_tokens(oracle, prompt)?;
    if prompt < ROWS + SOURCE_SUFFIX || ROWS >= context {
        return Err("Inkling split check needs a prompt of at least 6,004 tokens");
    }
    let rows = ROWS;
    let mut session = Session::load(checkpoint, context)?;
    session.set_batched_prefill(true, true)?;
    session.set_text_prefill_scope(true);
    if !session.runtime.source_text_active {
        return Err("Inkling split check requires the active source path");
    }
    let mut arms = Vec::new();
    let mut verdicts = serde_json::Map::new();
    for exact in [false, true] {
        let mut identical = true;
        for (class, patterns) in GROUPS.iter().enumerate() {
            let mut fingerprints = Vec::new();
            for &pattern in patterns.iter() {
                let ranges = split_ranges(rows, pattern);
                let widths = if class == 0 {
                    1025..=2048
                } else {
                    PackedPrefill::MIN_ROWS..=1024
                };
                if !ranges.iter().all(|range| widths.contains(&range.len())) {
                    return Err("Inkling split pattern leaves its width class");
                }
                session.runtime.exact_prefill = exact;
                session.reset();
                let ingested = ranges.iter().try_for_each(|range| session.ingest_many(&tokens[range.clone()]));
                session.runtime.exact_prefill = false;
                ingested?;
                let state = target_validation::state_digest(&session);
                let logits = target_validation::native_digest(&session.step(tokens[rows])?);
                arms.push(json!({
                    "exact_reference": exact, "width_class": class,
                    "commands": ranges.iter().map(ExactSizeIterator::len).collect::<Vec<_>>(),
                    "state_sha256": state, "next_logits_sha256": logits,
                }));
                fingerprints.push((state, logits));
            }
            identical &= fingerprints.windows(2).all(|pair| pair[0] == pair[1]);
        }
        verdicts.insert(
            if exact {
                "exact_identical"
            } else {
                "default_identical"
            }
            .into(),
            Value::Bool(identical),
        );
    }
    Ok(json!({
        "prompt_tokens": prompt,
        "context": context,
        "shared_rows": rows,
        "verdicts": verdicts,
        "runs": arms,
    }))
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    const SOURCE_ROWS: usize = Session::SOURCE_PREFILL_ROWS;

    fn lengths(ranges: &[Range<usize>]) -> Vec<usize> {
        ranges.iter().map(ExactSizeIterator::len).collect()
    }

    fn contiguous(
        ranges: &[Range<usize>],
        end: usize,
    ) -> bool {
        ranges.first().is_none_or(|first| first.start == 0)
            && ranges.windows(2).all(|pair| pair[0].end == pair[1].start)
            && ranges.last().is_none_or(|last| last.end == end)
    }

    #[stark_test]
    fn split_patterns_stay_in_the_packed_source_class() {
        for (pattern, widths) in [
            (&[2048][..], 1025..=2048),
            (&[1777, 1234, 1489, 1500], 1025..=2048),
            (&[16, 1000, 64, 65, 500, 355], PackedPrefill::MIN_ROWS..=1024),
            (&[333, 17, 674], PackedPrefill::MIN_ROWS..=1024),
        ] {
            for rows in [6000] {
                let ranges = split_ranges(rows, pattern);
                assert!(contiguous(&ranges, rows), "{rows}");
                assert!(ranges.iter().all(|range| widths.contains(&range.len())), "{rows} {:?}", lengths(&ranges));
            }
        }
    }

    #[stark_test]
    fn short_source_prompts_use_bulk_tail_and_seed() {
        for (prompt, bulk) in [(69, 65), (73, 69), (130, 126), (511, 507)] {
            let ranges = worker_ranges(prompt, true, SOURCE_ROWS);
            assert_eq!(lengths(&ranges), [bulk, 3], "{prompt}");
            assert!(contiguous(&ranges, prompt - 1), "{prompt}");
            assert!(replay_boundaries(&ranges, prompt, true, SOURCE_ROWS).is_empty(), "{prompt}");
        }
        // The exact reference engages the packed source kernels from 65 rows,
        // which N = 68 misses; the default path packs every command of 16 rows or more.
        assert_eq!(lengths(&worker_ranges(68, true, SOURCE_ROWS)), [64, 3]);
        assert_eq!(PackedPrefill::SOURCE_MIN_ROWS, 65);
    }

    #[stark_test]
    fn worker_admission_mirror_admits_every_nonempty_prompt() {
        assert!(!worker_admits_source(0));
        for prompt in [1, 4, 5, 68, 69, 511, 512, 4096] {
            assert!(worker_admits_source(prompt), "{prompt}");
        }
    }

    #[stark_test]
    fn stock_4k_schedule_is_unchanged() {
        let ranges = worker_ranges(4096, true, SOURCE_ROWS);
        assert_eq!(ranges, [0..2048, 2048..4092, 4092..4095]);
        assert_eq!(replay_boundaries(&ranges, 4096, true, SOURCE_ROWS), [2048]);
    }

    #[stark_test]
    fn long_source_prompts_cover_both_fixed_boundaries() {
        let ranges = worker_ranges(8800, true, SOURCE_ROWS);
        assert_eq!(lengths(&ranges), [2048, 2048, 2048, 2048, 604, 3]);
        assert!(contiguous(&ranges, 8799));
        assert_eq!(replay_boundaries(&ranges, 8800, true, SOURCE_ROWS), [4096, 8192]);

        let ranges = worker_ranges(16_000, true, SOURCE_ROWS);
        assert_eq!(lengths(&ranges), [2048, 2048, 2048, 2048, 2048, 2048, 2048, 1660, 3]);
        assert!(contiguous(&ranges, 15_999));
        assert_eq!(replay_boundaries(&ranges, 16_000, true, SOURCE_ROWS), [4096, 8192, 14_336]);

        let ranges = worker_ranges(8188, true, SOURCE_ROWS);
        assert_eq!(lengths(&ranges), [2048, 2048, 2048, 2040, 3]);
        assert_eq!(replay_boundaries(&ranges, 8188, true, SOURCE_ROWS), [4096, 6144]);

        let ranges = worker_ranges(4350, true, SOURCE_ROWS);
        assert_eq!(lengths(&ranges), [2048, 2048, 250, 3]);
        assert_eq!(replay_boundaries(&ranges, 4350, true, SOURCE_ROWS), [4096]);
    }

    #[stark_test]
    fn incumbent_schedule_ingests_all_but_the_seed() {
        let ranges = worker_ranges(68, false, 512);
        assert_eq!(ranges, [0..67]);
        assert!(replay_boundaries(&ranges, 68, false, 512).is_empty());
        let ranges = worker_ranges(8800, false, 512);
        assert!(contiguous(&ranges, 8799));
        assert!(ranges.iter().all(|range| range.len() <= 512));
        assert_eq!(replay_boundaries(&ranges, 8800, false, 512), [4096, 8192]);
        assert!(worker_ranges(0, false, 512).is_empty());
        assert!(worker_ranges(1, true, 512).is_empty());
    }
}
