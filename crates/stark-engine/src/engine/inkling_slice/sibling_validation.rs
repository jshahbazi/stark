// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only exactness checks for packed sibling verification.

use std::{fs, path::Path};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    trace_verify_validation::worker_prefill,
    transaction_validation::{greedy, same_logits},
    *,
};

#[derive(Clone, Copy)]
enum Scenario {
    SiblingFirst,
    SiblingFirstFour,
    SiblingSecond,
    PrimaryPath,
    RejectFirst,
    RejectSecond,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::SiblingFirst => "sibling_first_slot_three_rows",
            Self::SiblingFirstFour => "sibling_first_slot_four_rows",
            Self::SiblingSecond => "sibling_second_slot",
            Self::PrimaryPath => "primary_path",
            Self::RejectFirst => "reject_first_slot",
            Self::RejectSecond => "reject_second_slot",
        }
    }

    fn depth(self) -> usize {
        match self {
            Self::SiblingFirst | Self::SiblingFirstFour | Self::RejectFirst => 1,
            Self::SiblingSecond | Self::PrimaryPath | Self::RejectSecond => 2,
        }
    }
}

fn array_u32(value: &Value) -> Result<Vec<u32>> {
    value
        .as_array()
        .ok_or("invalid packed-sibling sidecar array")?
        .iter()
        .map(|item| {
            u32::try_from(item.as_u64().ok_or("invalid packed-sibling sidecar integer")?)
                .map_err(|_| "packed-sibling sidecar integer is out of range")
        })
        .collect()
}

fn select_offset(
    scenario: Scenario,
    primary: &[Vec<u32>],
    alternate: &[Vec<u32>],
    teacher: &[u32],
) -> Option<usize> {
    let enough = match scenario {
        Scenario::SiblingFirst | Scenario::RejectFirst => 1,
        _ => 2,
    };
    (0..primary.len().min(alternate.len())).find(|&i| {
        i + enough < teacher.len()
            && match scenario {
                Scenario::SiblingFirst | Scenario::SiblingFirstFour => {
                    primary[i][0] != teacher[i + 1] && alternate[i][0] == teacher[i + 1]
                },
                Scenario::SiblingSecond => {
                    primary[i][0] == teacher[i + 1]
                        && primary[i][1] != teacher[i + 2]
                        && alternate[i][1] == teacher[i + 2]
                },
                Scenario::PrimaryPath => primary[i][0] == teacher[i + 1] && primary[i][1] == teacher[i + 2],
                Scenario::RejectFirst => primary[i][0] != teacher[i + 1] && alternate[i][0] != teacher[i + 1],
                Scenario::RejectSecond => {
                    primary[i][0] == teacher[i + 1]
                        && primary[i][1] != teacher[i + 2]
                        && alternate[i][1] != teacher[i + 2]
                },
            }
    })
}

fn path_tokens(
    scenario: Scenario,
    offset: usize,
    primary: &[Vec<u32>],
    alternate: &[Vec<u32>],
    teacher: &[u32],
) -> Vec<u32> {
    match scenario {
        Scenario::SiblingFirst | Scenario::SiblingFirstFour => vec![teacher[offset], alternate[offset][0]],
        Scenario::SiblingSecond => vec![teacher[offset], primary[offset][0], alternate[offset][1]],
        Scenario::PrimaryPath => vec![teacher[offset], primary[offset][0], primary[offset][1]],
        Scenario::RejectFirst => vec![teacher[offset]],
        Scenario::RejectSecond => vec![teacher[offset], primary[offset][0]],
    }
}

fn packed_tokens(
    scenario: Scenario,
    offset: usize,
    primary: &[Vec<u32>],
    alternate: &[Vec<u32>],
    teacher: &[u32],
) -> Vec<u32> {
    match scenario {
        Scenario::SiblingFirst | Scenario::RejectFirst => {
            vec![teacher[offset], primary[offset][0], alternate[offset][0]]
        },
        Scenario::SiblingFirstFour => {
            vec![teacher[offset], primary[offset][0], primary[offset][1], alternate[offset][0]]
        },
        Scenario::SiblingSecond | Scenario::PrimaryPath | Scenario::RejectSecond => {
            vec![teacher[offset], primary[offset][0], primary[offset][1], alternate[offset][1]]
        },
    }
}

fn selected_logits_row(scenario: Scenario) -> usize {
    match scenario {
        Scenario::SiblingFirst => 2,
        Scenario::SiblingFirstFour | Scenario::SiblingSecond => 3,
        Scenario::PrimaryPath => 2,
        Scenario::RejectFirst => 0,
        Scenario::RejectSecond => 1,
    }
}

fn take_features(session: &Session) -> Result<Vec<u16>> {
    let mut result = Vec::with_capacity(DRAFT_FEATURE_WORDS);
    session.append_external_draft_features(1, &mut result)?;
    if result.len() != DRAFT_FEATURE_WORDS {
        return Err("Inkling sibling validation did not capture one draft row");
    }
    Ok(result)
}

fn run_scenario(
    session: &mut Session,
    prompt_base: &Snapshot,
    teacher: &[u32],
    primary: &[Vec<u32>],
    alternate: &[Vec<u32>],
    scenario: Scenario,
) -> Result<bool> {
    let offset = select_offset(scenario, primary, alternate, teacher)
        .ok_or("packed-sibling validation fixture lacks a required path")?;
    session.restore(prompt_base)?;
    for &token in &teacher[..offset] {
        session.step(token)?;
    }
    let base = session.snapshot()?;
    let path = path_tokens(scenario, offset, primary, alternate, teacher);
    let mut expected_features = Vec::with_capacity(path.len() * DRAFT_FEATURE_WORDS);
    let mut expected_logits = Vec::new();
    for &token in &path {
        expected_logits = session.step(token)?;
        expected_features.extend(take_features(session)?);
    }
    let expected_state = target_validation::state_digest(session);
    session.restore(&base)?;

    let depth = scenario.depth();
    let packet = packed_tokens(scenario, offset, primary, alternate, teacher);
    if !session.can_pack_sibling(packet.len()) {
        return Err("packed-sibling validation path is unsupported by this session");
    }
    let transaction = session.begin_packed_sibling(&packet, depth)?;
    let logits_exact = same_logits(transaction.logits(selected_logits_row(scenario))?, &expected_logits);
    match scenario {
        Scenario::SiblingFirst | Scenario::SiblingFirstFour | Scenario::SiblingSecond => {
            transaction.commit_sibling(depth)?;
        },
        Scenario::PrimaryPath => {
            transaction.commit(3)?;
        },
        Scenario::RejectFirst => {
            transaction.commit(1)?;
        },
        Scenario::RejectSecond => {
            transaction.commit(2)?;
        },
    }
    let state_exact = target_validation::state_digest(session) == expected_state;
    let mut actual_features = Vec::with_capacity(expected_features.len());
    session.append_external_draft_features(path.len(), &mut actual_features)?;
    let features_exact = actual_features == expected_features;

    // Continue from the committed transaction and compare its resulting state
    // with a serial replay from the same anchor.
    let continuation = greedy(&expected_logits);
    let actual_continuation = session.step(continuation)?;
    let actual_continuation_state = target_validation::state_digest(session);
    let expected_continuation = {
        session.restore(&base)?;
        for &token in &path {
            session.step(token)?;
        }
        session.step(continuation)?
    };
    let continuation_exact = same_logits(&actual_continuation, &expected_continuation);
    let continuation_state_exact = actual_continuation_state == target_validation::state_digest(session);
    Ok(logits_exact && state_exact && features_exact && continuation_exact && continuation_state_exact)
}

/// Verify packed sibling readouts, rollback/commit state, draft taps and live continuation.
/// The report contains only aggregate booleans and counts; fixture contents stay private.
pub fn measure_packed_sibling(
    checkpoint: &Path,
    trace_path: &Path,
    sidecar_path: &Path,
) -> Result<Value> {
    let trace_bytes = fs::read(trace_path).map_err(|_| "cannot read private sibling trace")?;
    let sidecar: Value =
        serde_json::from_slice(&fs::read(sidecar_path).map_err(|_| "cannot read private sibling sidecar")?)
            .map_err(|_| "invalid private sibling sidecar")?;
    let recorded_hash =
        sidecar.get("trace_sha256").and_then(Value::as_str).ok_or("sibling sidecar is missing its trace hash")?;
    let actual_hash = format!("{:x}", Sha256::digest(&trace_bytes));
    if recorded_hash != actual_hash {
        return Err("sibling sidecar is not bound to the supplied trace");
    }
    let trace = Tensors::open(trace_path)?;
    let prompt_len = trace.values::<u32>("prompt_len", "U32", &[1])?[0] as usize;
    let teacher_len = trace.values::<u32>("teacher_len", "U32", &[1])?[0] as usize;
    let tokens = trace.values::<u32>("tokens", "U32", &[prompt_len + teacher_len])?;
    let teacher = array_u32(sidecar.get("teacher").ok_or("sibling sidecar has no teacher sequence")?)?;
    if prompt_len == 0 || teacher.len() != teacher_len || tokens[prompt_len..] != teacher {
        return Err("sibling sidecar teacher does not match its private trace");
    }
    let primary = sidecar
        .get("primary")
        .and_then(Value::as_array)
        .ok_or("sibling sidecar has no primary rows")?
        .iter()
        .map(array_u32)
        .collect::<Result<Vec<_>>>()?;
    let alternate = sidecar
        .get("alternate")
        .and_then(Value::as_array)
        .ok_or("sibling sidecar has no alternate rows")?
        .iter()
        .map(array_u32)
        .collect::<Result<Vec<_>>>()?;
    if primary.iter().any(|row| row.len() != 3) || alternate.iter().any(|row| row.len() != 2) {
        return Err("sibling sidecar has invalid proposal geometry");
    }

    let mut session = Session::load(checkpoint, 4352)?;
    session.set_batched_prefill_width(true, true, 8192)?;
    session.set_external_draft(true)?;
    session.set_source_verifier(true)?;
    worker_prefill(&mut session, &tokens[..prompt_len])?;
    let prompt_base = session.snapshot()?;
    let scenarios = [
        Scenario::SiblingFirst,
        Scenario::SiblingFirstFour,
        Scenario::SiblingSecond,
        Scenario::PrimaryPath,
        Scenario::RejectFirst,
        Scenario::RejectSecond,
    ];
    let mut exact = 0;
    let mut checked = 0;
    let mut by_scenario = serde_json::Map::new();
    for scenario in scenarios {
        let valid = run_scenario(&mut session, &prompt_base, &teacher, &primary, &alternate, scenario)?;
        exact += usize::from(valid);
        checked += 1;
        by_scenario.insert(scenario.name().to_owned(), Value::Bool(valid));
    }
    Ok(json!({
        "schema": "inkling-packed-sibling-native-v1",
        "checks": checked,
        "exact_checks": exact,
        "all_exact": exact == checked,
        "scenarios": by_scenario,
    }))
}
