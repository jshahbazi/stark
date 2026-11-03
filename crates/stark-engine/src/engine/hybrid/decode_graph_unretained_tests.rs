// SPDX-License-Identifier: Apache-2.0
//! Explicit ownership for the opt-in GLM unretained-command transport.
use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn unretained_constructor_rejects_unowned_parallel_and_timing_transports() {
    for (parallel, phase, timing) in [(true, false, false), (false, true, false), (false, false, true)] {
        assert!(DecodeGraph::new_with_ownership(parallel, phase, timing, true).is_err());
    }
}

#[stark_test]
fn unretained_graph_owns_temporary_raw_external_operands_until_completion() {
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let output = graph.scratch(13).unwrap();
    let temporary = DecodeGraph::upload(&[1.25; 19]).unwrap().slice(3, 13).unwrap();
    let id = Retained::as_ptr(&temporary.buffer);
    graph.retain_external_buffer(&temporary.buffer);
    assert!(graph.external_resources.borrow().iter().any(|buffer| Retained::as_ptr(buffer) == id));
    let executor = executor().unwrap();
    let function = executor.library.new_function_with_name("hybrid_graph_copy").unwrap();
    // This test-local PSO remains owned until wait; production PSOs have
    // process-lifetime caches. All raw operand buffers are retained explicitly.
    let pipeline = executor.device.new_compute_pipeline_state_with_function(&function).unwrap();
    let encoder = graph.encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(Some(&temporary.buffer), temporary.offset, 0);
    encoder.set_buffer(Some(&output.buffer), output.offset, 1);
    let parameters = [13u32, 13];
    encoder.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 2);
    encoder.set_buffer(Some(&graph.failure.buffer), graph.failure.offset, 30);
    encoder.dispatch_threads(
        MTLSize {
            width: 13,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    drop(temporary);
    let pending = graph.submit(&[output.clone()]).unwrap();
    drop(output);
    assert_eq!(pending.wait().unwrap(), [vec![1.25; 13]]);
    drop(pipeline);
}

fn matrix(value: f32) -> MappedMatrix {
    let values = vec![value; 13 * 64];
    let bytes = bytemuck::cast_slice(&values);
    MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor().unwrap(), bytes).unwrap(),
            offset: 0,
            length: bytes.len(),
            registered_length: bytes.len(),
        },
        planes: None,
    }
}

#[stark_test]
fn unretained_graph_owns_multi_matrix_transitions_and_resource_boundaries() {
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let input = graph.upload_into(&[1.0; 128]).unwrap();
    graph.switch_dispatch(MTLDispatchType::CONCURRENT).unwrap();
    let output = {
        let operation = MappedOperation {
            matrices: vec![matrix(1.0), matrix(2.0)],
            inputs: &[],
            rows: 13,
            columns: 64,
            rows_per_batch: 0,
            format: MatrixFormat::Packed(0),
        };
        graph.mapped(&operation, &input).unwrap()
    };
    graph.switch_dispatch(MTLDispatchType::SERIAL).unwrap();
    graph.resource_boundary().unwrap();
    assert!(graph.retired_encoders.len() >= 3, "internal encoder owners were released early");
    let doubled = graph.add(&output, &output).unwrap();
    drop(input);
    drop(output);
    let actual = graph.submit(&[doubled]).unwrap().wait().unwrap().remove(0);
    assert_eq!(&actual[..13], &[128.0; 13]);
    assert_eq!(&actual[13..], &[256.0; 13]);
}

#[stark_test]
fn unretained_graph_preserves_overflow_failure_and_dropped_pending_drain() {
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let input = graph.upload_into(&[f32::MAX; 64]).unwrap();
    let output = graph
        .mapped(
            &MappedOperation {
                matrices: vec![matrix(2.0)],
                inputs: &[],
                rows: 13,
                columns: 64,
                rows_per_batch: 0,
                format: MatrixFormat::Packed(0),
            },
            &input,
        )
        .unwrap();
    assert!(graph.submit(&[output]).unwrap().wait().is_err());
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let source = graph.upload_into(&[0.5; 13]).unwrap();
    let target = DecodeGraph::zeros(13).unwrap();
    graph.copy_into(&source, &target).unwrap();
    graph.resource_boundary().unwrap();
    let command = graph.command.clone();
    let pending = graph.submit(&[]).unwrap();
    drop(source);
    drop(pending);
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: pending Drop fenced the completed writer; target is separately owned.
    let values = unsafe { std::slice::from_raw_parts(target.buffer.contents().as_ptr().cast::<f32>(), 13) };
    assert_eq!(values, &[0.5; 13]);
}
