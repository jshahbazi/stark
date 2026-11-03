// SPDX-License-Identifier: Apache-2.0
//! Exact-view retention and overwrite ordering for selective finite scans.

use stark_engine_macros::stark_test;

use super::*;

fn graph(concurrent: bool) -> DecodeGraph {
    let mut graph = DecodeGraph::new_with_timing(concurrent, false, false).unwrap();
    graph.batch_finite = true;
    graph.finite_selective = true;
    graph.serial_projection_levels = false;
    graph
}

#[stark_test]
fn selective_finite_preserves_masked_poison_and_unrelated_queues() {
    for concurrent in [false, true] {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut graph = graph(concurrent);
            let target = DecodeGraph::zeros(13).unwrap();
            // SAFETY: newly owned test storage has no pending GPU users.
            unsafe {
                target.buffer.contents().as_ptr().cast::<f32>().add(12).write(poison);
            }
            let unrelated = DecodeGraph::upload(&[2.0; 17]).unwrap();
            graph.check_finite(&target).unwrap();
            graph.check_finite(&unrelated.slice(0, 7).unwrap()).unwrap();
            graph.begin_projection_level().unwrap();
            graph.check_finite(&target.slice(6, 7).unwrap()).unwrap();
            graph.check_finite(&unrelated.slice(7, 10).unwrap()).unwrap();
            let finite = DecodeGraph::upload(&[0.0; 13]).unwrap();
            graph.copy_into(&finite, &target).unwrap();
            assert_eq!(graph.pending_finite.len(), 1, "unrelated pending view was flushed");
            assert_eq!(graph.projection_checks.len(), 1, "unrelated projection view was flushed");
            assert!(std::ptr::eq(&*graph.pending_finite[0].buffer, &*unrelated.buffer));
            graph.end_projection_level().unwrap();
            assert_eq!(graph.pending_finite.len(), 2);
            graph.resource_boundary().unwrap();
            assert!(graph.finish(&[target]).is_err(), "overwrite concealed a checked nonfinite");
        }
    }
}

#[stark_test]
fn selective_finite_skips_nonoverlap_rewrites_and_does_not_rescan_consumed_views() {
    profile::reset();
    profile::enable(true);
    let mut graph = graph(true);
    let allocation = DecodeGraph::zeros(33).unwrap();
    // NaN gaps expose accidental range widening while checked views stay finite.
    unsafe {
        let p = allocation.buffer.contents().as_ptr().cast::<f32>();
        for at in 0..33 {
            p.add(at).write(f32::NAN);
        }
        for at in (0..7).chain(20..33) {
            p.add(at).write(1.0);
        }
    }
    let a = allocation.slice(0, 7).unwrap();
    let b = allocation.slice(20, 13).unwrap();
    graph.check_finite(&a).unwrap();
    graph.check_finite(&b).unwrap();
    let fresh = allocation.slice(10, 3).unwrap();
    graph.copy_into(&DecodeGraph::upload(&[4.0; 3]).unwrap(), &fresh).unwrap();
    assert_eq!(graph.pending_finite.len(), 2);
    assert!(!profile::snapshot().contains_key("native.metal.graph.finite_batches"));
    graph.copy_into(&DecodeGraph::upload(&[5.0; 7]).unwrap(), &a).unwrap();
    assert_eq!(graph.pending_finite.len(), 1);
    assert_eq!(graph.pending_finite[0].offset, b.offset);
    assert_eq!(profile::snapshot()["native.metal.graph.finite_checks_batched"].bytes, 1);
    graph.copy_into(&DecodeGraph::upload(&[6.0; 7]).unwrap(), &a).unwrap();
    assert_eq!(profile::snapshot()["native.metal.graph.finite_checks_batched"].bytes, 1);
    let values = graph.finish(&[a, b]).unwrap();
    let counts = profile::snapshot();
    profile::enable(false);
    assert_eq!(values, [vec![6.0; 7], vec![1.0; 13]]);
    assert_eq!(counts["native.metal.graph.finite_checks_batched"].bytes, 2);
    assert_eq!(counts["native.metal.graph.finite_batches"].calls, 2);
}

#[stark_test]
fn selective_finite_bounds_combined_pending_and_projection_metadata_and_recovers() {
    let mut graph = graph(false);
    let value = DecodeGraph::upload(&[3.0]).unwrap();
    for _ in 0..8192 {
        graph.check_finite(&value).unwrap();
    }
    graph.begin_projection_level().unwrap();
    for _ in 0..8192 {
        graph.check_finite(&value).unwrap();
    }
    assert_eq!(graph.pending_finite.len() + graph.projection_checks.len(), 16_384);
    graph.check_finite(&value).unwrap();
    assert!(graph.pending_finite.is_empty());
    assert_eq!(graph.projection_checks.len(), 1);
    graph.end_projection_level().unwrap();
    assert_eq!(graph.finish(&[value]).unwrap(), [vec![3.0]]);
    let mut recovered = self::graph(false);
    let value = DecodeGraph::upload(&[1.0]).unwrap();
    recovered.check_finite(&value).unwrap();
    assert_eq!(recovered.finish(&[value]).unwrap(), [vec![1.0]]);
}
