// SPDX-License-Identifier: Apache-2.0
//! Serialized actual-artifact combined restoration and admission gate.
//! All tokens, complete readouts and activations remain private in RAM.

use std::path::PathBuf;

use serde_json::json;
use stark_engine_macros::stark_test;

use super::*;
use crate::engine::hybrid::{Forward, Model};

fn head_bits(mut output: HeadOutput) -> (Vec<u32>, Vec<u32>) {
    assert!(output.stages.is_empty(), "combined_actual_trace_disabled");
    let logits = output.logits.take().unwrap_or_else(|| panic!("combined_actual_head_readout"));
    let wide = output.wide.take().unwrap_or_else(|| panic!("combined_actual_head_wide"));
    assert!(logits.len() == 248_320 && logits.iter().all(|value| value.is_finite()), "combined_actual_head_values");
    (logits.into_iter().map(f32::to_bits).collect(), wide_bits(&wide))
}

fn wide_bits(wide: &DeviceVector) -> Vec<u32> {
    let mut rows = new_graph()
        .and_then(|graph| graph.finish(&[wide.clone()]))
        .unwrap_or_else(|_| panic!("combined_actual_wide_read"));
    assert!(rows.len() == 1, "combined_actual_wide_inventory");
    let values = rows.pop().unwrap_or_else(|| panic!("combined_actual_wide_missing"));
    assert!(values.len() == 10_240 && values.iter().all(|value| value.is_finite()), "combined_actual_wide_values");
    values.into_iter().map(f32::to_bits).collect()
}

#[stark_test]
#[ignore]
fn edge0_mtp_actual_combined_restore_rebind_and_resource_lease() {
    assert!(
        std::env::var("STARK_EDGE0_REFERENCE").ok().as_deref() == Some("1")
            && std::env::var("STARK_EDGE0_METAL").ok().as_deref() == Some("1")
            && std::env::var("STARK_EDGE0_RESIDENT").ok().as_deref() != Some("1"),
        "combined_actual_policy"
    );
    let path =
        PathBuf::from(std::env::var_os("STARK_EDGE0_SOURCE_PATH").unwrap_or_else(|| panic!("combined_actual_source")));
    let mut model = Model::load(&path).unwrap_or_else(|_| panic!("combined_actual_load"));
    let encoding = model
        .tokenizer()
        .encode("A quiet morning brings clear light through the window. ".repeat(8), false)
        .unwrap_or_else(|_| panic!("combined_actual_encode"));
    let tokens = encoding.get_ids().get(..33).unwrap_or_else(|| panic!("combined_actual_corpus"));
    let Forward::Qwen(target) = &mut model.forward else {
        panic!("combined_actual_backend");
    };
    let policy = checkpoint::StagingPolicy::Edge0MetalSingleStepV1;
    let mut head = Head::for_target(target).unwrap_or_else(|_| panic!("combined_actual_head_admission"));
    assert!(Head::for_target(target).is_err(), "combined_actual_concurrent_head_refusal");
    let mut pending = None;
    for &token in &tokens[..31] {
        if let Some(wide) = &pending {
            head.target_catch_up(target, wide, token, false).unwrap_or_else(|_| panic!("combined_actual_front"));
        }
        let (_, wide) = target.step_with_wide(token).unwrap_or_else(|_| panic!("combined_actual_prompt_step"));
        pending = Some(wide);
    }
    let pending = pending.unwrap_or_else(|| panic!("combined_actual_pending"));
    let pending_bits = wide_bits(&pending.values);
    let mut saved =
        CombinedCheckpoint::capture(target, &mut head, &pending).unwrap_or_else(|_| panic!("combined_actual_capture"));
    let first_head = head_bits(
        head.target_step(target, &pending, tokens[31], false).unwrap_or_else(|_| panic!("combined_actual_first_head")),
    );
    let (first_target, next_pending) =
        target.step_with_wide(tokens[31]).unwrap_or_else(|_| panic!("combined_actual_first_target"));
    let first_target = first_target.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let second_head = head_bits(
        head.target_step(target, &next_pending, tokens[32], false)
            .unwrap_or_else(|_| panic!("combined_actual_second_head")),
    );
    let (second_target, _) =
        target.step_with_wide(tokens[32]).unwrap_or_else(|_| panic!("combined_actual_second_target"));
    let second_target = second_target.into_iter().map(f32::to_bits).collect::<Vec<_>>();

    let restored = saved.restore(target, &mut head).unwrap_or_else(|_| panic!("combined_actual_restore"));
    assert!(target.position == 31 && head.next_cell() == 30, "combined_actual_restored_boundary");
    assert!(wide_bits(&restored.values) == pending_bits, "combined_actual_pending_bits");
    assert!(
        !target
            .checkpoint_state_tag_matches(&pending.tag, policy)
            .unwrap_or_else(|_| panic!("combined_actual_stale_tag")),
        "combined_actual_old_wide_stale"
    );
    assert!(
        head_bits(
            head.target_step(target, &restored, tokens[31], false)
                .unwrap_or_else(|_| panic!("combined_actual_first_replay_head"))
        ) == first_head,
        "combined_actual_first_head_bits"
    );
    let (row, next_replayed) =
        target.step_with_wide(tokens[31]).unwrap_or_else(|_| panic!("combined_actual_first_replay_target"));
    assert!(row.into_iter().map(f32::to_bits).eq(first_target), "combined_actual_first_target_bits");
    assert!(
        head_bits(
            head.target_step(target, &next_replayed, tokens[32], false)
                .unwrap_or_else(|_| panic!("combined_actual_second_replay_head"))
        ) == second_head,
        "combined_actual_second_head_bits"
    );
    let (row, _) = target.step_with_wide(tokens[32]).unwrap_or_else(|_| panic!("combined_actual_second_replay_target"));
    assert!(row.into_iter().map(f32::to_bits).eq(second_target), "combined_actual_second_target_bits");
    let retained = saved.restore(target, &mut head).unwrap_or_else(|_| panic!("combined_actual_repeat_restore"));
    assert!(
        !target
            .checkpoint_state_tag_matches(&restored.tag, policy)
            .unwrap_or_else(|_| panic!("combined_actual_repeat_tag")),
        "combined_actual_previous_restore_stale"
    );

    let original_wide = saved.wide[0];
    saved.wide[0] = f32::NAN;
    assert!(saved.restore(target, &mut head).is_err(), "combined_actual_malformed_refusal");
    assert!(target.position == 0 && head.is_fenced(), "combined_actual_joint_failure_fence");
    saved.wide[0] = original_wide;
    assert!(saved.restore(target, &mut head).is_err(), "combined_actual_failure_expires_prior_handles");
    head.rebind_reset(target).unwrap_or_else(|_| panic!("combined_actual_reset_rebind"));
    assert!(!head.is_fenced() && head.next_cell() == 0, "combined_actual_reset_rebound");
    assert!(saved.restore(target, &mut head).is_err(), "combined_actual_recovered_session_stale_refusal");
    assert!(target.position == 0 && head.is_fenced(), "combined_actual_recovered_session_joint_fence");
    head.rebind_reset(target).unwrap_or_else(|_| panic!("combined_actual_recovered_session_rebind"));
    assert!(
        head.target_step(target, &pending, tokens[31], false).is_err() && head.is_fenced(),
        "combined_actual_stale_wide_execution_refusal"
    );
    head.rebind_reset(target).unwrap_or_else(|_| panic!("combined_actual_stale_wide_rebind"));
    // Checkpoints and restored wide owners independently retain admission.
    drop(head);
    assert!(Head::for_target(target).is_err(), "combined_actual_snapshot_holds_admission");
    drop(saved);
    drop(restored);
    assert!(Head::for_target(target).is_err(), "combined_actual_wide_holds_admission");
    drop(retained);
    drop(pending);
    drop(next_pending);
    drop(next_replayed);
    let mut head = Head::for_target(target).unwrap_or_else(|_| panic!("combined_actual_recreate_without_reload"));
    let (_, fresh) = target.step_with_wide(tokens[0]).unwrap_or_else(|_| panic!("combined_actual_fresh_step"));
    head.target_catch_up(target, &fresh, tokens[1], false).unwrap_or_else(|_| panic!("combined_actual_fresh_binding"));
    head.fence_target(target);
    head.rebind_reset(target).unwrap_or_else(|_| panic!("combined_actual_second_reset_rebind"));
    println!(
        "{}",
        json!({"actual_combined_restore":true,"target_advances":36,
        "target_paired_exact_rows":2,"head_paired_exact_rows":2,"pending_wide_exact_rows":1,
        "repeated_restore":true,"malformed_joint_fence":true,"stale_handles_rejected":true,
        "snapshot_and_wide_lease_retention":true,"recreated_without_model_reload":true,
        "reset_rebind":true,"mtp_sampling":false})
    );
}
