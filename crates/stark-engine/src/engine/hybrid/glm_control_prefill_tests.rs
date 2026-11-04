// SPDX-License-Identifier: Apache-2.0

use stark_engine_macros::stark_test;

use super::*;

fn make_control(
    hidden: usize,
    streams: usize,
    chunks: usize,
    simd: bool,
    fused: bool,
) -> Control {
    let rows = (2 + streams) * streams;
    let mut value = Control::new(
        ControlGeometry {
            hidden,
            streams,
            iterations: 5,
            eps: 1e-6,
            rms_eps: 1e-5,
        },
        &(0..rows * hidden * streams).map(|i| ((i * 13 % 127) as f32 - 63.0) * 0.0001).collect::<Vec<_>>(),
        &(0..rows).map(|i| (i as f32 - 7.0) * 0.01).collect::<Vec<_>>(),
        &[0.03, -0.02, 0.01],
        &(0..hidden).map(|i| 1.0 + (i % 11) as f32 * 0.003).collect::<Vec<_>>(),
    )
    .unwrap();
    // Explicit controls avoid environment mutation while exercising every
    // projection and mixing policy with the same graph RMS flags on both sides.
    value.projection_chunks = chunks;
    value.projection = DecodeGraph::zeros(rows * chunks).unwrap();
    value.simd_mix = simd;
    value.rms_project = fused;
    value.fused_parallel = fused;
    value
}

fn fixture(
    control: &Control,
    tokens: usize,
    batched: bool,
    unretained: bool,
) -> (DecodeGraph, Vec<DeviceVector>) {
    fixture_with_stress(control, tokens, batched, unretained, false)
}

fn fixture_with_stress(
    control: &Control,
    tokens: usize,
    batched: bool,
    unretained: bool,
    stress: bool,
) -> (DecodeGraph, Vec<DeviceVector>) {
    let hidden = control.input.len;
    let columns = control.parameters[3] as usize;
    // Nonzero vector offsets and a final row that is not dispatch-width aligned
    // catch cross-token addressing and padded-grid writes.
    let residual = DecodeGraph::upload(
        &(0..tokens * columns + 14)
            .map(|i| {
                // Keep the large-value fallback local to alternating tokens.
                let scale = if stress && (i.saturating_sub(7) / columns) % 2 == 1 {
                    1e16
                } else {
                    0.7
                };
                (i as f32 * 0.017).sin() * scale
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
    .slice(7, tokens * columns)
    .unwrap();
    let branch =
        DecodeGraph::upload(&(0..tokens * hidden + 10).map(|i| (i as f32 * 0.013).cos() * 0.1).collect::<Vec<_>>())
            .unwrap()
            .slice(5, tokens * hidden)
            .unwrap();
    let mut graph = if unretained {
        DecodeGraph::new_unretained()
    } else {
        DecodeGraph::new()
    }
    .unwrap();
    let normalized = graph.scratch(tokens * hidden).unwrap();
    let destination = graph.scratch(tokens * columns).unwrap();
    let mixes = if batched {
        let mix = control.pre_batch(&mut graph, &residual, &normalized, tokens).unwrap();
        control.post_batch(&mut graph, &residual, &branch, &mix, &destination, tokens).unwrap();
        mix
    } else {
        let mix = graph.scratch(tokens * control.mix.len).unwrap();
        let mut snapshots = Vec::new();
        for token in 0..tokens {
            snapshots.push(
                control
                    .pre_batch_row(
                        &mut graph,
                        &residual.slice(token * columns, columns).unwrap(),
                        &normalized.slice(token * hidden, hidden).unwrap(),
                    )
                    .unwrap(),
            );
        }
        for token in (0..tokens).rev() {
            graph.copy_into(&snapshots[token], &mix.slice(token * control.mix.len, control.mix.len).unwrap()).unwrap();
            control
                .post_batch_row(
                    &mut graph,
                    &residual.slice(token * columns, columns).unwrap(),
                    &branch.slice(token * hidden, hidden).unwrap(),
                    &snapshots[token],
                    &destination.slice(token * columns, columns).unwrap(),
                )
                .unwrap();
        }
        mix
    };
    (graph, vec![normalized, mixes, destination])
}

fn assert_bits(
    actual: &[Vec<f32>],
    expected: &[Vec<f32>],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()), "batched mHC changed row bits");
    }
}

#[stark_test]
fn control_prefill_batch_matches_selected_row_schedules_with_token_and_column_tails() {
    for streams in [1, 4, 6, 8] {
        let mut schedules = vec![(1, false)];
        if streams <= 4 {
            schedules.extend([(1, true), (8, true), (16, true), (32, true)]);
        }
        for (chunks, simd) in schedules {
            let control = make_control(35, streams, chunks, simd, false);
            for tokens in [1, 3, 17] {
                let (graph, outputs) = fixture(&control, tokens, false, false);
                let expected = graph.finish(&outputs).unwrap();
                let (graph, outputs) = fixture(&control, tokens, true, false);
                assert_bits(&graph.finish(&outputs).unwrap(), &expected);
            }
        }
    }
    let control = make_control(19, 4, 32, true, false);
    let (graph, outputs) = fixture(&control, 128, false, false);
    let expected = graph.finish(&outputs).unwrap();
    let (graph, outputs) = fixture(&control, 128, true, false);
    assert_bits(&graph.finish(&outputs).unwrap(), &expected);
}

#[stark_test]
fn control_prefill_batch_matches_rms_projection_and_optional_fusion() {
    for fused in [false, true] {
        let mut control = make_control(4096, 4, 32, true, false);
        control.rms_project = !control.ordered_rms;
        control.fused_parallel = fused && !control.ordered_rms;
        for stress in [false, true] {
            let (graph, outputs) = fixture_with_stress(&control, 3, false, false, stress);
            let expected = graph.finish(&outputs).unwrap();
            let (graph, outputs) = fixture_with_stress(&control, 3, true, false, stress);
            assert_bits(&graph.finish(&outputs).unwrap(), &expected);
        }
    }
}

#[stark_test]
fn control_prefill_batch_queued_unretained_graph_owns_buffers_after_control_drop() {
    let control = make_control(35, 4, 16, true, false);
    let (graph, outputs) = fixture(&control, 7, false, false);
    let expected = graph.finish(&outputs).unwrap();
    drop(control);
    let pending = objc2::rc::autoreleasepool(|_| {
        let control = make_control(35, 4, 16, true, false);
        let (graph, outputs) = fixture(&control, 7, true, true);
        graph.validate_queued_submission().unwrap();
        graph.submit(&outputs).unwrap()
    });
    assert_bits(&pending.wait().unwrap(), &expected);
}

#[stark_test]
fn control_prefill_batch_rejects_invalid_spans_tokens_and_destination_aliases() {
    let control = make_control(35, 4, 8, true, false);
    let mut graph = DecodeGraph::new().unwrap();
    let residual = DecodeGraph::zeros(3 * 140).unwrap();
    let normalized = graph.scratch(3 * 35).unwrap();
    let branch = DecodeGraph::zeros(3 * 35).unwrap();
    let mix = graph.scratch(3 * 24).unwrap();
    let destination = graph.scratch(3 * 140).unwrap();
    assert!(control.pre_batch(&mut graph, &residual, &normalized, 0).is_err());
    assert!(control.pre_batch(&mut graph, &residual, &normalized, 129).is_err());
    assert!(control.pre_batch(&mut graph, &residual, &normalized, 2).is_err());
    assert!(control.pre_batch(&mut graph, &residual, &residual.slice(0, 105).unwrap(), 3).is_err());
    let invalid = DeviceVector {
        buffer: normalized.buffer.clone(),
        offset: normalized.buffer.length(),
        len: 105,
    };
    assert!(control.pre_batch(&mut graph, &residual, &invalid, 3).is_err());
    assert!(control.post_batch(&mut graph, &residual, &branch, &mix, &residual, 3).is_err());
    assert!(control.post_batch(&mut graph, &residual, &branch, &control.mix, &destination, 3).is_err());
    graph.finish(&[]).unwrap();
}

#[stark_test]
fn control_prefill_batch_preserves_nonfinite_obligations_and_fresh_graph_recovery() {
    let control = make_control(35, 4, 32, true, false);
    for post in [false, true] {
        let mut graph = DecodeGraph::new().unwrap();
        let residual = DecodeGraph::zeros(3 * 140).unwrap();
        let normalized = graph.scratch(3 * 35).unwrap();
        let destination = graph.scratch(3 * 140).unwrap();
        let overwritten = if post {
            &destination
        } else {
            &normalized
        };
        unsafe {
            overwritten
                .buffer
                .contents()
                .as_ptr()
                .cast::<f32>()
                .add(overwritten.offset / 4 + overwritten.len - 1)
                .write(f32::NAN)
        };
        graph.check_finite(overwritten).unwrap();
        let mix = control.pre_batch(&mut graph, &residual, &normalized, 3).unwrap();
        let branch = DecodeGraph::zeros(3 * 35).unwrap();
        control.post_batch(&mut graph, &residual, &branch, &mix, &destination, 3).unwrap();
        assert!(graph.finish(&[normalized, destination]).is_err(), "batch overwrite hid a finite obligation");
    }
    for poison_branch in [false, true] {
        let mut graph = DecodeGraph::new().unwrap();
        let residual = DecodeGraph::zeros(3 * 140).unwrap();
        let branch = DecodeGraph::zeros(3 * 35).unwrap();
        // Uploads reject nonfinite host data. Inject into the retained shared
        // input before enqueue to exercise the device operators' checks.
        let (poisoned, index, value) = if poison_branch {
            (&branch, 2 * 35 + 34, f32::NAN)
        } else {
            (&residual, 2 * 140 + 137, f32::INFINITY)
        };
        unsafe { poisoned.buffer.contents().as_ptr().cast::<f32>().add(poisoned.offset / 4 + index).write(value) };
        let normalized = graph.scratch(3 * 35).unwrap();
        let destination = graph.scratch(3 * 140).unwrap();
        let mix = control.pre_batch(&mut graph, &residual, &normalized, 3).unwrap();
        control.post_batch(&mut graph, &residual, &branch, &mix, &destination, 3).unwrap();
        assert!(graph.finish(&[normalized, destination]).is_err(), "batch discarded a nonfinite row");
    }
    let (graph, outputs) = fixture(&control, 3, true, false);
    assert!(graph.finish(&outputs).is_ok());
}
