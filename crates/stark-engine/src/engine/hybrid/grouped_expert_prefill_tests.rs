// SPDX-License-Identifier: Apache-2.0
use stark_engine_macros::stark_test;

use super::*;

fn ids(values: &[u32]) -> DeviceVector {
    let vector = DecodeGraph::upload(&vec![0.0; values.len()]).unwrap();
    // No command is submitted yet; preserve the router's raw u32 storage.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), vector.buffer.contents().as_ptr().cast::<u32>(), values.len());
    }
    vector
}

fn matrix(
    kind: u32,
    experts: usize,
    rows: usize,
    columns: usize,
) -> MappedMatrix {
    let block_bytes = weights::block_layout(kind).unwrap().1;
    let length = experts * rows * (columns / 256) * block_bytes;
    let buffer = executor().unwrap().device.new_buffer(length + 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
    let bytes = unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u8>(), length + 4) };
    let mut seed = 17u32;
    for byte in &mut bytes[2..2 + length] {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *byte = (seed >> 24) as u8;
    }
    for block in bytes[2..2 + length].chunks_exact_mut(block_bytes) {
        block[..2].copy_from_slice(&0x1800u16.to_le_bytes());
    }
    MappedMatrix {
        codes: MappedSlice {
            buffer,
            offset: 2,
            length,
            registered_length: length + 4,
        },
        planes: None,
    }
}

#[expect(clippy::too_many_arguments, reason = "reference retains the existing per-token indexed operation explicitly")]
fn reference(
    matrix: &MappedMatrix,
    format: MatrixFormat,
    shape: [usize; 3],
    input: &DeviceVector,
    indices: &[u32],
    tokens: usize,
    selected: usize,
    batched: bool,
) -> Vec<f32> {
    let [experts, rows, columns] = shape;
    let executor = executor().unwrap();
    let pipeline = executor
        .cached_pipeline(format, iq_bound_factor_name(format).unwrap(), &executor.iq_bound_factor_indexed_pipelines)
        .unwrap();
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let output = graph.scratch(tokens * selected * rows).unwrap();
    for token in 0..tokens {
        let route_ids = ids(&indices[token * selected..(token + 1) * selected]);
        let row = input
            .slice(
                token
                    * columns
                    * if batched {
                        selected
                    } else {
                        1
                    },
                columns
                    * if batched {
                        selected
                    } else {
                        1
                    },
            )
            .unwrap();
        let destination = output.slice(token * selected * rows, selected * rows).unwrap();
        let bounds = graph
            .iq_input_bounds(
                &row,
                columns,
                if batched {
                    selected
                } else {
                    1
                },
            )
            .unwrap();
        graph.order_dispatch();
        graph.encoding.set_compute_pipeline_state(&pipeline);
        for (index, buffer, offset) in [
            (0, &matrix.codes.buffer, matrix.codes.offset),
            (1, &row.buffer, row.offset),
            (2, &destination.buffer, destination.offset),
            (6, &route_ids.buffer, route_ids.offset),
            (7, &bounds.buffer, bounds.offset),
        ] {
            graph.encoding.set_buffer(Some(buffer), offset, index);
            graph.resources.push(buffer.clone());
        }
        let p = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
        graph.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        graph.encoding.set_buffer(Some(&graph.failure.buffer), graph.failure.offset, 30);
        graph.encoding.dispatch_threadgroups(
            MTLSize {
                width: rows.div_ceil(8),
                height: selected,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    }
    graph.resources.push(matrix.codes.buffer.clone());
    graph.finish(&[output]).unwrap().remove(0)
}

#[stark_test]
fn grouped_expert_prefill_geometry_and_capacity_bounds() {
    assert!(route_geometry(288, 128, 8));
    assert!(route_geometry(1, 128, 1));
    for (experts, tokens, selected) in [(0, 1, 1), (288, 0, 8), (288, 129, 8), (288, 128, 9), (2, 1, 3), (2049, 1, 1)] {
        assert!(!route_geometry(experts, tokens, selected));
    }
}

#[stark_test]
fn grouped_expert_prefill_rejects_forged_weight_spans_before_encoding() {
    let shape = [5, 9, 2048];
    let valid = matrix(17, shape[0], shape[1], shape[2]);
    let mut graph = DecodeGraph::new_unretained().unwrap();
    // Shape-only plan: malformed weights must be rejected before any GPU
    // command could read this deliberately uninitialized descriptor storage.
    let plan = GroupedExpertRoutes {
        counts: graph.scratch(5).unwrap(),
        lists: graph.scratch(20).unwrap(),
        jobs: graph.scratch(21).unwrap(),
        experts: 5,
        tokens: 2,
        selected: 2,
    };
    let input = graph.scratch(2 * 2048).unwrap();
    let resources = graph.resources.len();
    let scratch = graph.scratch_backing_bytes();
    let mut descriptors = Vec::new();
    let mut past_end = valid.clone();
    past_end.codes.offset = 6;
    descriptors.push(past_end);
    let mut overflow = valid.clone();
    overflow.codes.offset = usize::MAX - 1;
    descriptors.push(overflow);
    let mut unaligned = valid.clone();
    unaligned.codes.offset = 3;
    descriptors.push(unaligned);
    let mut planes = valid.clone();
    planes.planes = Some((valid.codes.clone(), valid.codes.clone()));
    descriptors.push(planes);
    let mut truncated = valid.clone();
    truncated.codes.buffer = executor().unwrap().device.new_buffer(8, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
    descriptors.push(truncated);
    for descriptor in descriptors {
        assert!(
            graph
                .mapped_expert_grouped_inner(&descriptor, MatrixFormat::Packed(17), shape, &plan, &input, false)
                .is_err(),
            "malformed expert weight span was admitted"
        );
        assert_eq!(graph.resources.len(), resources, "invalid expert descriptor bound resources");
        assert_eq!(graph.scratch_backing_bytes(), scratch, "invalid expert descriptor allocated scratch");
    }
}

#[stark_test]
fn grouped_expert_prefill_index_copy_preserves_raw_bits_and_rejects_aliases() {
    let values = [17, 0, 1, 287, 0x8000_0000, 0x7fc0_0123, u32::MAX, 19];
    let source = ids(&values);
    let destination = ids(&[11; 8]);
    let mut graph = DecodeGraph::new_unretained().unwrap();
    graph.copy_expert_indices(&source.slice(1, 6).unwrap(), &destination.slice(1, 6).unwrap()).unwrap();
    assert!(graph.copy_expert_indices(&source.slice(0, 4).unwrap(), &source.slice(2, 4).unwrap()).is_err());
    assert!(graph.copy_expert_indices(&source, &source).is_err());
    assert!(graph.copy_expert_indices(&source, &destination.slice(0, 7).unwrap()).is_err());
    graph.finish(&[]).unwrap();
    let actual = unsafe { std::slice::from_raw_parts(destination.buffer.contents().as_ptr().cast::<u32>(), 8) };
    assert!(actual[0] == 11 && actual[7] == 11 && actual[1..7] == values[1..7], "routing bit copy changed storage");
}

#[stark_test]
fn grouped_expert_prefill_matches_selected_iq_rows_and_mixed_envelopes() {
    for kind in [17, 18] {
        for columns in [2048, 4096] {
            for (tokens, selected) in [(1, 1), (3, 2), (17, 3)] {
                let shape = [5, 9, columns];
                let matrix = matrix(kind, shape[0], shape[1], columns);
                let indices: Vec<_> = (0..tokens * selected).map(|i| ((i * 3 + i / 4) % 5) as u32).collect();
                for batched in [false, true] {
                    for extreme in [false, true] {
                        let mut values: Vec<_> = (0..columns
                            * if batched {
                                tokens * selected
                            } else {
                                tokens
                            })
                            .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
                            .collect();
                        if extreme {
                            values[columns / 2] = 1.0e21;
                        }
                        let input = DecodeGraph::upload(&values).unwrap();
                        let expected = reference(
                            &matrix,
                            MatrixFormat::Packed(kind),
                            shape,
                            &input,
                            &indices,
                            tokens,
                            selected,
                            batched,
                        );
                        let mut graph = DecodeGraph::new_unretained().unwrap();
                        let plan = graph
                            .group_expert_routes_inner(&ids(&indices), shape[0], tokens, selected)
                            .unwrap()
                            .unwrap();
                        let result = graph
                            .mapped_expert_grouped_inner(
                                &matrix,
                                MatrixFormat::Packed(kind),
                                shape,
                                &plan,
                                &input,
                                batched,
                            )
                            .unwrap();
                        let actual = graph.finish(&[result]).unwrap().remove(0);
                        assert!(
                            actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "grouped experts changed selected F32 row bits"
                        );
                    }
                }
            }
        }
    }
}

#[stark_test]
fn grouped_expert_prefill_rejects_invalid_ids_and_masked_nonfinite_rows() {
    let shape = [5, 9, 2048];
    let matrix = matrix(17, shape[0], shape[1], shape[2]);
    for invalid in [5, u32::MAX, 0x7f80_0000] {
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let plan = graph.group_expert_routes_inner(&ids(&[0, invalid, 1, 2]), 5, 2, 2).unwrap().unwrap();
        let input = DecodeGraph::upload(&vec![0.0; 4096]).unwrap();
        let result =
            graph.mapped_expert_grouped_inner(&matrix, MatrixFormat::Packed(17), shape, &plan, &input, false).unwrap();
        assert!(graph.finish(&[result]).is_err(), "invalid grouped expert ID escaped status");
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let plan = graph.group_expert_routes_inner(&ids(&[0, 0, 0, 0]), 5, 2, 2).unwrap().unwrap();
        let input = DecodeGraph::upload(&vec![0.0; 4096]).unwrap();
        unsafe {
            input.buffer.contents().as_ptr().cast::<f32>().add(2048).write(value);
        }
        let result =
            graph.mapped_expert_grouped_inner(&matrix, MatrixFormat::Packed(17), shape, &plan, &input, false).unwrap();
        let masked = graph.scratch(result.len).unwrap();
        graph.copy_into(&DecodeGraph::upload(&vec![0.0; result.len]).unwrap(), &masked).unwrap();
        assert!(graph.finish(&[masked]).is_err(), "masked grouped expert poison escaped status");
    }
}

#[stark_test]
fn grouped_expert_prefill_unretained_ownership_and_abandoned_work_are_safe() {
    let pending = {
        let shape = [5, 9, 2048];
        let matrix = matrix(18, shape[0], shape[1], shape[2]);
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let input = DecodeGraph::upload(&vec![0.125; 2 * 2048]).unwrap();
        let plan = graph.group_expert_routes_inner(&ids(&[2, 2, 2, 2]), 5, 2, 2).unwrap().unwrap();
        let result =
            graph.mapped_expert_grouped_inner(&matrix, MatrixFormat::Packed(18), shape, &plan, &input, false).unwrap();
        graph.submit(&[result]).unwrap()
    };
    let actual = pending.wait().unwrap();
    assert!(actual[0].iter().all(|value| value.is_finite()));
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let plan = graph.group_expert_routes_inner(&ids(&[1, 1]), 5, 1, 2).unwrap().unwrap();
    drop(plan);
    drop(graph); // No submission may escape this cancellation point.
}
