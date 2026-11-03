// SPDX-License-Identifier: Apache-2.0
//! Ownership, completion and failure contracts for submitted decode graphs.

use stark_engine_macros::stark_test;

use super::*;

struct RestoreProfile(bool);

impl Drop for RestoreProfile {
    fn drop(&mut self) {
        profile::enable(self.0);
    }
}

fn capture_profile() -> RestoreProfile {
    let restore = RestoreProfile(profile::enabled());
    profile::reset();
    profile::enable(true);
    restore
}

#[stark_test]
fn submitted_graph_retains_resources_and_output_views_without_waiting() {
    let _profile = capture_profile();
    for concurrent in [false, true] {
        profile::reset();
        let mut graph = DecodeGraph::new_with_timing(concurrent, false, false).unwrap();
        let input = graph.upload_into(&[1.0, -2.0, 3.0, 0.0, 5.0]).unwrap();
        let output = graph.add(&input, &input).unwrap();
        let pending = graph.submit(&[output.slice(1, 3).unwrap()]).unwrap();
        drop(input);
        drop(output);
        assert!(pending.submitted);
        assert!(!pending.completed);
        assert!(pending.graph.encoding_ended);
        assert!(!pending.graph.resources.is_empty());
        assert_eq!(profile::snapshot()["native.metal.submit"].calls, 1);
        assert!(!profile::snapshot().contains_key("native.metal.wait"));

        // Encoding the next command may retain submitted outputs. Discarding
        // it has no submission or completion side effect on the first owner.
        let mut discarded = DecodeGraph::new_with_timing(concurrent, false, false).unwrap();
        discarded.add(&pending.outputs[0], &pending.outputs[0]).unwrap();
        drop(discarded);
        assert_eq!(profile::snapshot()["native.metal.submit"].calls, 1);
        assert!(!profile::snapshot().contains_key("native.metal.wait"));
        assert_eq!(pending.wait().unwrap(), [vec![-4.0, 6.0, 0.0]]);
        let counts = profile::snapshot();
        assert_eq!(counts["native.metal.submit"].calls, 1);
        assert_eq!(counts["native.metal.wait"].calls, 1);
        assert_eq!(counts["native.metal.gpu"].calls, 1);
    }
}

#[stark_test]
fn submitted_graph_reports_sticky_failures_and_recovers() {
    let _profile = capture_profile();
    for batched in [false, true] {
        profile::reset();
        let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
        graph.batch_finite = batched;
        let poison = DecodeGraph::zeros(1031).unwrap();
        // SAFETY: new owned test storage has no pending GPU users.
        unsafe {
            poison.buffer.contents().as_ptr().cast::<f32>().add(1030).write(f32::NAN);
        }
        graph.check_finite(&poison).unwrap();
        let finite = graph.upload_into(&[7.0]).unwrap();
        let pending = graph.submit(&[finite]).unwrap();
        assert!(pending.wait().is_err(), "submission concealed an unreturned nonfinite span");
        assert_eq!(profile::snapshot()["native.metal.wait"].calls, 1);
        let mut recovered = DecodeGraph::new_with_timing(false, false, false).unwrap();
        let value = recovered.upload_into(&[3.0]).unwrap();
        assert_eq!(recovered.finish(&[value]).unwrap(), [vec![3.0]]);
        assert_eq!(profile::snapshot()["native.metal.wait"].calls, 2);
    }
}

#[stark_test]
fn dropped_pending_graph_drains_before_releasing_owners() {
    let _profile = capture_profile();
    let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
    let input = graph.upload_into(&vec![0.75; 4097]).unwrap();
    let output = DecodeGraph::zeros(4097).unwrap();
    graph.copy_into(&input, &output).unwrap();
    let command = graph.command.clone();
    let pending = graph.submit(&[output.clone()]).unwrap();
    drop(input);
    drop(pending);
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert_eq!(profile::snapshot()["native.metal.wait"].calls, 1);
    // SAFETY: pending Drop returned only after the writer completed; this
    // separately retained view still owns the initialized shared allocation.
    let values = unsafe { std::slice::from_raw_parts(output.buffer.contents().as_ptr().cast::<f32>(), output.len) };
    assert!(values.iter().all(|&v| v == 0.75));
}

#[stark_test]
fn invalid_outputs_and_unfinished_levels_never_submit() {
    let _profile = capture_profile();
    let graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
    let command = graph.command.clone();
    let allocation = DecodeGraph::zeros(1).unwrap();
    let invalid = DeviceVector {
        buffer: allocation.buffer.clone(),
        offset: 4,
        len: 1,
    };
    assert!(graph.submit(&[invalid]).is_err());
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
    assert!(!profile::snapshot().contains_key("native.metal.submit"));
    assert!(!profile::snapshot().contains_key("native.metal.wait"));

    let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
    graph.serial_projection_levels = false;
    graph.begin_projection_level().unwrap();
    let command = graph.command.clone();
    assert!(graph.submit(&[]).is_err());
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
    assert!(!profile::snapshot().contains_key("native.metal.submit"));
}

#[stark_test]
fn submitted_graphs_preserve_same_queue_dependencies() {
    let mut first = DecodeGraph::new_with_timing(false, false, false).unwrap();
    let input = first.upload_into(&[1.0, -2.0, 3.0]).unwrap();
    let doubled = first.add(&input, &input).unwrap();
    let pending_first = first.submit(&[doubled.clone()]).unwrap();
    let mut second = DecodeGraph::new_with_timing(false, false, false).unwrap();
    let tripled = second.add(&doubled, &input).unwrap();
    let pending_second = second.submit(&[tripled]).unwrap();
    // Both graphs were committed in queue order; waiting on the later owner
    // first must still observe the earlier command's writes.
    assert_eq!(pending_second.wait().unwrap(), [vec![3.0, -6.0, 9.0]]);
    assert_eq!(pending_first.wait().unwrap(), [vec![2.0, -4.0, 6.0]]);
}

#[stark_test]
fn finish_and_explicit_submission_preserve_phase_boundaries_and_bits() {
    for phase_timing in [false, true] {
        let run = |explicit| {
            let mut graph = DecodeGraph::new_with_timing(false, phase_timing, false).unwrap();
            graph.batch_finite = true;
            let input = graph.upload_into(&[0.25, -0.75, 1.25]).unwrap();
            let first = graph.add(&input, &input).unwrap();
            graph.check_finite(&first).unwrap();
            graph.diagnostic_checkpoint("native.test.async.phase").unwrap();
            let output = graph.add(&first, &input).unwrap();
            if explicit {
                graph.submit(&[output]).unwrap().wait().unwrap()
            } else {
                graph.finish(&[output]).unwrap()
            }
        };
        let original = run(false);
        let explicit = run(true);
        assert!(original.iter().flatten().zip(explicit.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[stark_test]
fn counter_checkpoints_retain_finite_checks_while_phase_submissions_flush() {
    let _profile = capture_profile();
    for phase in [false, true] {
        for poisoned in [false, true] {
            profile::reset();
            let mut graph = DecodeGraph::new_with_timing(false, phase, !phase).unwrap();
            graph.batch_finite = true;
            graph.finite_selective = true;
            let checked = DecodeGraph::zeros(1031).unwrap();
            if poisoned {
                // SAFETY: freshly owned synthetic storage has no GPU users.
                unsafe {
                    checked.buffer.contents().as_ptr().cast::<f32>().add(1030).write(f32::NAN);
                }
            }
            graph.check_finite(&checked).unwrap();
            graph.diagnostic_checkpoint("native.test.finite.checkpoint").unwrap();
            assert_eq!(
                graph.pending_finite.len(),
                if phase {
                    0
                } else {
                    1
                }
            );
            assert_eq!(profile::snapshot().contains_key("native.metal.graph.finite_batches"), phase);
            let output = graph.upload_into(&[7.0]).unwrap();
            let actual = graph.finish(&[output]);
            assert_eq!(actual.is_err(), poisoned, "checkpoint lost a finite obligation");
            assert_eq!(profile::snapshot()["native.metal.graph.finite_checks_batched"].bytes, 1);
        }
    }
}

#[stark_test]
fn pending_poll_and_blocking_modes_preserve_outputs_failures_and_drain() {
    let _profile = capture_profile();
    for poll in [false, true] {
        let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
        let input = graph.upload_into(&[0.25; 9]).unwrap();
        let output = graph.add(&input, &input).unwrap();
        let mut pending = graph.submit(&[output]).unwrap();
        pending.poll_wait = poll;
        assert_eq!(pending.wait().unwrap(), [vec![0.5; 9]]);

        let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
        let poison = DecodeGraph::zeros(1).unwrap();
        // SAFETY: freshly owned test storage has no pending GPU users.
        unsafe {
            poison.buffer.contents().as_ptr().cast::<f32>().write(f32::NAN);
        }
        graph.check_finite(&poison).unwrap();
        let mut pending = graph.submit(&[]).unwrap();
        pending.poll_wait = poll;
        assert!(pending.wait().is_err());

        profile::reset();
        let mut graph = DecodeGraph::new_with_timing(false, false, false).unwrap();
        let source = graph.upload_into(&[0.75; 9]).unwrap();
        let target = DecodeGraph::zeros(9).unwrap();
        graph.copy_into(&source, &target).unwrap();
        let command = graph.command.clone();
        let mut pending = graph.submit(&[]).unwrap();
        pending.poll_wait = poll;
        drop(pending);
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(profile::snapshot()["native.metal.wait"].calls, 1);
    }
}
