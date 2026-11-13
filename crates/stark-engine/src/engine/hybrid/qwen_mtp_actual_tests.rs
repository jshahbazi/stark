// SPDX-License-Identifier: Apache-2.0
//! Bounded actual-artifact head/target-wide lifecycle gate. One loaded model,
//! virtual weights, ordinary fixed text, private RAM-only IDs and readouts.
//! This does not qualify sampling, combined request restoration or full context.

use std::path::PathBuf;

use serde_json::json;
use stark_engine_macros::stark_test;

use super::*;
use crate::engine::hybrid::{Forward, Model};

#[stark_test]
// Requires the registered prepared artifact and a serialized Metal run under
// the external 64-GiB available / 48-GiB owned-RSS / 600-second resource gate.
#[ignore]
fn edge0_mtp_actual_head_and_target_wide_lifecycle() {
    // Inspect complete head rows privately. Returning the compact device wide
    // owner also supplies the next draft cell without a host reconstruction.
    fn output_bits(mut output: HeadOutput) -> (DeviceVector, Vec<u32>, Vec<u32>) {
        let logits = output.logits.take().unwrap_or_else(|| panic!("actual_head_readout_missing"));
        assert!(logits.len() == 248_320 && logits.iter().all(|value| value.is_finite()), "actual_head_readout_values");
        assert!(output.stages.is_empty(), "actual_head_trace_disabled");
        let wide = output.wide.take().unwrap_or_else(|| panic!("actual_head_wide_missing"));
        assert!(wide.len == 10_240, "actual_head_wide_shape");
        let mut rows = new_graph()
            .and_then(|graph| graph.finish(&[wide.clone()]))
            .unwrap_or_else(|_| panic!("actual_head_wide_read"));
        assert!(rows.len() == 1, "actual_head_wide_rows");
        let values = rows.pop().unwrap_or_else(|| panic!("actual_head_wide_row_missing"));
        assert!(values.len() == 10_240 && values.iter().all(|value| value.is_finite()), "actual_head_wide_values");
        (wide, logits.into_iter().map(f32::to_bits).collect(), values.into_iter().map(f32::to_bits).collect())
    }

    assert!(
        std::env::var("STARK_EDGE0_REFERENCE").ok().as_deref() == Some("1")
            && std::env::var("STARK_EDGE0_METAL").ok().as_deref() == Some("1")
            && std::env::var("STARK_EDGE0_RESIDENT").ok().as_deref() != Some("1"),
        "actual_head_execution_policy"
    );
    let path = PathBuf::from(
        std::env::var_os("STARK_EDGE0_SOURCE_PATH").unwrap_or_else(|| panic!("actual_head_source_required")),
    );
    let mut model = Model::load(&path).unwrap_or_else(|_| panic!("actual_head_load"));
    let text = "A quiet morning brings clear light through the window. ".repeat(8);
    let encoding = model.tokenizer().encode(text, false).unwrap_or_else(|_| panic!("actual_head_encode"));
    let tokens = encoding.get_ids().get(..36).unwrap_or_else(|| panic!("actual_head_corpus_length"));
    let Forward::Qwen(target) = &mut model.forward else {
        panic!("actual_head_backend");
    };
    assert!(target.geometry.edge0 && target.geometry.edge0_metal && target.graph_enabled, "actual_head_target_policy");

    let mut target_advances = 0;
    let mut target_complete_readout_rows = 0;
    let mut target_paired_exact_rows = 0;
    let mut head_front_only_cells = 0;
    let mut head_full_readout_rows = 0;
    let mut head_paired_exact_rows = 0;
    let mut head_wide_paired_exact_rows = 0;
    target
        .ingest_prefix(&tokens[..32], || {
            target_advances += 1;
            Ok(())
        })
        .unwrap_or_else(|_| panic!("actual_head_reference_ingest"));
    let mut target_reference = Vec::with_capacity(4);
    for &token in &tokens[32..] {
        let row = target.step(token).unwrap_or_else(|_| panic!("actual_head_reference_step"));
        assert!(row.len() == 248_320 && row.iter().all(|value| value.is_finite()), "actual_head_reference_values");
        target_reference.push(row.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        target_advances += 1;
        target_complete_readout_rows += 1;
    }
    target.reset();
    let mut head = Head::for_target(target).unwrap_or_else(|_| panic!("actual_head_admission"));
    let mut saved_wide = None;
    for (row, &token) in tokens.iter().enumerate() {
        let (logits, wide) = target.step_with_wide(token).unwrap_or_else(|_| panic!("actual_head_wide_step"));
        target_advances += 1;
        target_complete_readout_rows += 1;
        assert!(
            logits.len() == 248_320
                && logits.iter().all(|value| value.is_finite())
                && wide.row == row
                && wide.values.len == 10_240,
            "actual_head_target_wide_values"
        );
        if row >= 32 {
            assert!(
                logits.iter().map(|value| value.to_bits()).eq(target_reference[row - 32].iter().copied()),
                "actual_head_target_wide_exact"
            );
            target_paired_exact_rows += 1;
        }
        if row < 31 {
            let front = head
                .target_catch_up(target, &wide, tokens[row + 1], false)
                .unwrap_or_else(|_| panic!("actual_head_front_cell"));
            assert!(
                front.wide.is_none()
                    && front.logits.is_none()
                    && front.routed_ids.is_empty()
                    && front.stages.is_empty(),
                "actual_head_front_only"
            );
            head_front_only_cells += 1;
        } else if row == 31 {
            saved_wide = Some(wide);
        }
    }
    let wide31 = saved_wide.unwrap_or_else(|| panic!("actual_head_saved_wide"));
    assert!(target.position == 36 && head.next_cell() == 31 && !head.is_fenced(), "actual_head_prefill_boundary");
    let head_checkpoint = head.capture_checkpoint().unwrap_or_else(|_| panic!("actual_head_capture"));
    let first =
        head.target_step(target, &wide31, tokens[32], false).unwrap_or_else(|_| panic!("actual_head_first_full"));
    let (first_wide, first_logits, first_bits) = output_bits(first);
    head_full_readout_rows += 1;
    let second = head
        .step(
            &ActualWeights {
                weights: &target.weights,
                embedding: &target.embedding,
                readout: &target.readout,
            },
            32,
            &first_wide,
            tokens[33],
            false,
        )
        .unwrap_or_else(|_| panic!("actual_head_chain_full"));
    let (_, second_logits, second_bits) = output_bits(second);
    head_full_readout_rows += 1;
    drop(first_wide);
    assert!(head.next_cell() == 33, "actual_head_chain_boundary");

    head.restore_checkpoint(&head_checkpoint).unwrap_or_else(|_| panic!("actual_head_replay_restore"));
    let first =
        head.target_step(target, &wide31, tokens[32], false).unwrap_or_else(|_| panic!("actual_head_replay_first"));
    let (first_wide, logits, bits) = output_bits(first);
    head_full_readout_rows += 1;
    assert!(logits == first_logits, "actual_head_first_readout_exact");
    assert!(bits == first_bits, "actual_head_first_wide_exact");
    head_paired_exact_rows += 1;
    head_wide_paired_exact_rows += 1;
    let second = head
        .step(
            &ActualWeights {
                weights: &target.weights,
                embedding: &target.embedding,
                readout: &target.readout,
            },
            32,
            &first_wide,
            tokens[33],
            false,
        )
        .unwrap_or_else(|_| panic!("actual_head_replay_chain"));
    let (_, logits, bits) = output_bits(second);
    head_full_readout_rows += 1;
    assert!(logits == second_logits, "actual_head_chain_readout_exact");
    assert!(bits == second_bits, "actual_head_chain_wide_exact");
    head_paired_exact_rows += 1;
    head_wide_paired_exact_rows += 1;
    drop(first_wide);

    let policy = checkpoint::StagingPolicy::Edge0MetalSingleStepV1;
    let target_checkpoint = target.capture_checkpoint(policy).unwrap_or_else(|_| panic!("actual_head_target_capture"));
    target.restore_checkpoint(&target_checkpoint, policy).unwrap_or_else(|_| panic!("actual_head_target_restore"));
    head.restore_checkpoint(&head_checkpoint).unwrap_or_else(|_| panic!("actual_head_branch_head_restore"));
    assert!(
        target.position == 36 && head.next_cell() == 31 && wide31.row == 31 && !head.is_fenced(),
        "actual_head_branch_causal_boundary"
    );
    assert!(
        !target
            .checkpoint_state_tag_matches(&wide31.tag, policy)
            .unwrap_or_else(|_| panic!("actual_head_branch_tag_check")),
        "actual_head_branch_tag_stale"
    );
    assert!(
        head.target_step(target, &wide31, tokens[32], false).is_err() && head.is_fenced(),
        "actual_head_branch_refusal"
    );
    assert!(head.restore_checkpoint(&head_checkpoint).is_err() && head.is_fenced(), "actual_head_old_handle_refusal");
    assert!(
        target_advances == 72
            && target_complete_readout_rows == 40
            && head_front_only_cells == 31
            && head_full_readout_rows == 4
            && target_paired_exact_rows == 4
            && head_paired_exact_rows == 2
            && head_wide_paired_exact_rows == 2,
        "actual_head_gate_counts"
    );
    println!(
        "{}",
        json!({
            "target_advances":target_advances, "target_complete_readout_rows":target_complete_readout_rows,
            "head_front_only_cells":head_front_only_cells, "head_full_readout_rows":head_full_readout_rows,
            "head_successful_cells":head_front_only_cells+head_full_readout_rows,
            "target_paired_exact_rows":target_paired_exact_rows, "head_paired_exact_rows":head_paired_exact_rows,
            "head_wide_paired_exact_rows":head_wide_paired_exact_rows, "stale_branch_rejected":true,
            "actual_artifact_forward":true, "mtp_sampling":false
        })
    );
}
