// SPDX-License-Identifier: Apache-2.0
//! Legacy-queue dependency ordering and tracked-resource admission.

use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn queued_graphs_preserve_tracked_state_dependencies_without_host_waits() {
    let heap = ScratchHeap::new(8 * 1024 * 1024).unwrap();
    let state = DecodeGraph::zeros(1031).unwrap();
    let mut first = DecodeGraph::new_with_scratch_heap(&heap, true).unwrap();
    let input = first.upload_into(&vec![1.25; 1031]).unwrap();
    first.copy_into(&input, &state).unwrap();
    first.validate_queued_submission().unwrap();
    let pending_first = first.submit(&[]).unwrap();

    let mut second = DecodeGraph::new_with_scratch_heap(&heap, true).unwrap();
    let doubled = second.add(&state, &state).unwrap();
    second.copy_into(&doubled, &state).unwrap();
    second.validate_queued_submission().unwrap();
    let pending_second = second.submit(&[state.clone()]).unwrap();
    drop(input);
    drop(doubled);
    drop(heap);
    // Both commands have been committed on the same legacy queue. The host
    // reads shared state only after the tail fence, never after just the head.
    pending_first.wait().unwrap();
    assert_eq!(pending_second.wait().unwrap(), [vec![2.5; 1031]]);
}

#[stark_test]
fn queued_graph_admission_rejects_unaudited_or_untracked_resources() {
    let retained = DecodeGraph::new_with_ownership(false, false, false, false).unwrap();
    assert!(retained.validate_queued_submission().is_err());
    let mut graph = DecodeGraph::new_with_ownership(false, false, false, true).unwrap();
    graph.serial_projection_levels = false;
    graph.begin_projection_level().unwrap();
    assert!(graph.validate_queued_submission().is_err());
    graph.end_projection_level().unwrap();
    graph.validate_queued_submission().unwrap();

    for external in [false, true] {
        let mut graph = DecodeGraph::new_with_ownership(false, false, false, true).unwrap();
        let buffer = executor()
            .unwrap()
            .device
            .new_buffer(4, MTLResourceOptions::STORAGE_MODE_SHARED | MTLResourceOptions::HAZARD_TRACKING_MODE_UNTRACKED)
            .unwrap();
        // Fresh owned storage is initialized before any submission.
        unsafe { buffer.contents().as_ptr().cast::<f32>().write(1.0) };
        if external {
            graph.retain_external_buffer(&buffer);
        } else {
            let value = DeviceVector {
                buffer,
                offset: 0,
                len: 1,
            };
            graph.add(&value, &value).unwrap();
        }
        assert!(graph.validate_queued_submission().is_err());
    }
}
