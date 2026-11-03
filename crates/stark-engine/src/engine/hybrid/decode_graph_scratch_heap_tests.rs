// SPDX-License-Identifier: Apache-2.0
//! Non-aliasable heap scratch ownership, exhaustion and completion contracts.
use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn scratch_heap_two_live_graphs_and_escaped_views_keep_backing_alive() {
    for unretained in [false, true] {
        let heap = ScratchHeap::new(8 * 1024 * 1024).unwrap();
        let id = Retained::as_ptr(&heap.heap);
        let mut first = DecodeGraph::new_with_scratch_heap(&heap, unretained).unwrap();
        let x = first.upload_into(&[1.0; 1031]).unwrap();
        let a = first.add(&x, &x).unwrap();
        let escaped = a.clone();
        let pending_a = first.submit(&[a]).unwrap();
        drop(x);
        let mut second = DecodeGraph::new_with_scratch_heap(&heap, unretained).unwrap();
        let y = second.upload_into(&[3.0; 1031]).unwrap();
        let b = second.add(&y, &y).unwrap();
        let pending_b = second.submit(&[b]).unwrap();
        drop(y);
        drop(heap);
        assert_eq!(pending_b.wait().unwrap(), [vec![6.0; 1031]]);
        assert_eq!(pending_a.wait().unwrap(), [vec![2.0; 1031]]);
        // Both graph owners and the caller's heap owner are gone. Apple
        // specifies that this live buffer alone strongly retains its heap.
        assert_eq!(Retained::as_ptr(&metal::MTLResource::heap(&*escaped.buffer).unwrap()), id);
        assert_eq!(metal::MTLResource::storage_mode(&*escaped.buffer), metal::MTLStorageMode::Shared);
        assert_eq!(metal::MTLResource::hazard_tracking_mode(&*escaped.buffer), metal::MTLHazardTrackingMode::Tracked);
        let mut consumer = DecodeGraph::new().unwrap();
        let doubled = consumer.add(&escaped, &escaped).unwrap();
        assert_eq!(consumer.finish(&[doubled]).unwrap(), [vec![4.0; 1031]]);
    }
}

#[stark_test]
fn scratch_heap_exhaustion_has_no_device_fallback_and_reuses_only_released_resources() {
    assert!(ScratchHeap::new(0).is_err());
    assert!(ScratchHeap::new(2 * 512 * 1024 * 1024 + 1).is_err());
    let capacity = 8 * 1024 * 1024;
    let heap = ScratchHeap::new(capacity).unwrap();
    let mut graph = DecodeGraph::new_with_scratch_heap(&heap, false).unwrap();
    let escaped = graph.upload_into(&[1.0]).unwrap();
    let before = graph.scratch_bytes;
    assert!(graph.scratch(capacity / 4).is_err(), "heap exhaustion silently allocated outside its capacity");
    assert_eq!(graph.scratch_bytes, before);
    drop(graph);
    assert!(heap.allocate(capacity).is_err(), "live escaped buffer became aliasable");
    drop(escaped);
    objc2::rc::autoreleasepool(|_| {
        let entire = heap.allocate(capacity).unwrap();
        assert_eq!(entire.length(), capacity);
    });
    for _ in 0..3 {
        let mut graph = DecodeGraph::new_with_scratch_heap(&heap, false).unwrap();
        let x = graph.upload_into(&[2.0; 13]).unwrap();
        let output = graph.add(&x, &x).unwrap();
        assert_eq!(graph.finish(&[output]).unwrap(), [vec![4.0; 13]]);
    }
    for (parallel, phase, timing) in [(true, false, false), (false, true, false), (false, false, true)] {
        assert!(DecodeGraph::new_with_scratch_policy(parallel, phase, timing, false, Some(heap.clone())).is_err());
    }
}

#[stark_test]
fn scratch_heap_preserves_nonfinite_status_reset_resource_boundaries_and_drop_drain() {
    for unretained in [false, true] {
        let heap = ScratchHeap::new(8 * 1024 * 1024).unwrap();
        let mut bad = DecodeGraph::new_with_scratch_heap(&heap, unretained).unwrap();
        bad.batch_finite = true;
        let poison = bad.upload_into(&[0.0; 1031]).unwrap();
        // SAFETY: this freshly allocated span is initialized before submit.
        unsafe {
            poison.buffer.contents().as_ptr().cast::<u8>().add(poison.offset).cast::<f32>().add(1030).write(f32::NAN);
        }
        bad.check_finite(&poison).unwrap();
        bad.resource_boundary().unwrap();
        assert!(bad.finish(&[]).is_err());
        drop(poison);
        let mut recovered = DecodeGraph::new_with_scratch_heap(&heap, unretained).unwrap();
        let source = recovered.upload_into(&[0.5; 13]).unwrap();
        let output = recovered.add(&source, &source).unwrap();
        assert_eq!(recovered.finish(&[output]).unwrap(), [vec![1.0; 13]]);
        drop(source);
        let mut abandoned = DecodeGraph::new_with_scratch_heap(&heap, unretained).unwrap();
        let source = abandoned.upload_into(&[0.75; 13]).unwrap();
        let target = abandoned.scratch(13).unwrap();
        abandoned.copy_into(&source, &target).unwrap();
        let command = abandoned.command.clone();
        let pending = abandoned.submit(&[]).unwrap();
        drop(source);
        drop(pending);
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let mut consumer = DecodeGraph::new().unwrap();
        let doubled = consumer.add(&target, &target).unwrap();
        assert_eq!(consumer.finish(&[doubled]).unwrap(), [vec![1.5; 13]]);
    }
}
