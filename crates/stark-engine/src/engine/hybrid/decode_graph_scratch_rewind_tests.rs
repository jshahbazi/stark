// SPDX-License-Identifier: Apache-2.0
//! Model-free checks for serial scratch checkpoint ownership and reuse.

use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn scratch_rewind_reuses_later_slabs_and_preserves_pre_checkpoint_vectors() {
    let mut graph = DecodeGraph::new().unwrap();
    let persistent = graph.upload_into(&[2.0]).unwrap();
    let checkpoint = graph.scratch_checkpoint().unwrap();
    let before_bytes = graph.scratch_bytes;

    // Each request is larger than the first slab and forces a separate arena.
    let first = graph.scratch(300_000).unwrap();
    let second = graph.scratch(300_000).unwrap();
    let first_id = Retained::as_ptr(&first.buffer) as usize;
    let first_offset = first.offset;
    let second_id = Retained::as_ptr(&second.buffer) as usize;
    let second_offset = second.offset;
    let allocated_bytes = graph.scratch_bytes;
    assert!(allocated_bytes > before_bytes);
    drop(first);
    drop(second);

    graph.scratch_rewind(&checkpoint).unwrap();
    let first_again = graph.scratch(300_000).unwrap();
    let second_again = graph.scratch(300_000).unwrap();
    assert_eq!(Retained::as_ptr(&first_again.buffer) as usize, first_id);
    assert_eq!(first_again.offset, first_offset);
    assert_eq!(Retained::as_ptr(&second_again.buffer) as usize, second_id);
    assert_eq!(second_again.offset, second_offset);
    assert_eq!(graph.scratch_bytes, allocated_bytes, "rewind allocated more backing");
    drop(first_again);
    drop(second_again);

    graph.scratch_rewind(&checkpoint).unwrap();
    let output = graph.add(&persistent, &persistent).unwrap();
    assert_eq!(graph.finish(&[output]).unwrap(), [vec![4.0]]);
}

#[stark_test]
fn scratch_rewind_flushes_finite_reads_before_recycled_gpu_writes() {
    let mut graph = DecodeGraph::new().unwrap();
    graph.batch_finite = true;
    let input = DecodeGraph::upload(&[0.0]).unwrap();
    let checkpoint = graph.scratch_checkpoint().unwrap();
    let pending_check = graph.scratch(1).unwrap();
    // SAFETY: this fresh scratch span is initialized before its queued scan.
    unsafe {
        pending_check.buffer.contents().as_ptr().cast::<u8>().add(pending_check.offset).cast::<f32>().write(f32::NAN);
    }
    graph.check_finite(&pending_check).unwrap();
    drop(pending_check);

    graph.scratch_rewind(&checkpoint).unwrap();
    let output = graph.add(&input, &input).unwrap();
    assert!(graph.finish(&[output]).is_err(), "recycled zero write erased a pending finite failure");
}

#[stark_test]
fn scratch_checkpoint_rejects_concurrent_timed_and_projection_scopes() {
    let mut concurrent = DecodeGraph::new_parallel().unwrap();
    assert!(concurrent.scratch_checkpoint().is_err());

    let mut timed = DecodeGraph::new_with_timing(false, false, true).unwrap();
    assert!(timed.scratch_checkpoint().is_err());

    let mut projection = DecodeGraph::new().unwrap();
    projection.serial_projection_levels = false;
    projection.begin_projection_level().unwrap();
    assert!(projection.scratch_checkpoint().is_err());
}

#[stark_test]
fn scratch_checkpoint_rejects_host_uploads_after_scope_starts() {
    let mut graph = DecodeGraph::new().unwrap();
    let checkpoint = graph.scratch_checkpoint().unwrap();
    assert!(graph.upload_into(&[3.0]).is_err());
    graph.scratch_rewind(&checkpoint).unwrap();
    assert!(graph.upload_into(&[4.0]).is_err());
}

#[stark_test]
fn scratch_checkpoint_is_bound_to_its_original_graph() {
    let mut owner = DecodeGraph::new().unwrap();
    let checkpoint = owner.scratch_checkpoint().unwrap();
    let mut other = DecodeGraph::new().unwrap();
    assert!(other.scratch_rewind(&checkpoint).is_err());
}

#[stark_test]
fn scratch_rewind_rejects_a_checkpoint_from_a_discarded_future_scope() {
    let mut graph = DecodeGraph::new().unwrap();
    let initial = graph.scratch_checkpoint().unwrap();
    let _temporary = graph.scratch(1).unwrap();
    let future = graph.scratch_checkpoint().unwrap();
    graph.scratch_rewind(&initial).unwrap();
    assert!(graph.scratch_rewind(&future).is_err());
}
