// SPDX-License-Identifier: Apache-2.0
//! Single-encoder transport parity across dependent and independent operators.

use stark_engine_macros::stark_test;

use super::*;

fn test_matrix(values: &[f32]) -> MappedMatrix {
    let bytes = bytemuck::cast_slice(values);
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
fn single_encoder_preserves_projection_levels_indexed_branches_and_dependent_operators() {
    let values =
        |len: usize, salt: usize| (0..len).map(|i| (((i * 19 + salt) % 67) as f32 - 33.0) * 0.017).collect::<Vec<_>>();
    let input = DecodeGraph::upload(&values(64, 3)).unwrap();
    let norm = DecodeGraph::upload(&vec![1.1; 64]).unwrap();
    let projection = test_matrix(&values(13 * 64, 5));
    let shared = test_matrix(&values(64 * 13, 7));
    let bank = test_matrix(&values(3 * 64 * 13, 11));
    let indices = DecodeGraph::upload(&[f32::from_bits(2), 0.0]).unwrap();
    let probabilities = DecodeGraph::upload(&[0.3, 0.7]).unwrap();
    let shared_gate = DecodeGraph::upload(&[0.4]).unwrap();
    let mut expected = None;
    for (single_encoder, serial_levels) in [(false, false), (true, false), (true, true)] {
        let mut graph = DecodeGraph::new_with_encoder(single_encoder).unwrap();
        graph.serial_projection_levels = serial_levels;
        let normalized = graph.rms_norm(&input, &norm, 64, 1e-6).unwrap();
        graph.begin_projection_level().unwrap();
        let mut projected = Vec::new();
        for _ in 0..2 {
            projected.push(
                graph
                    .mapped(
                        &MappedOperation {
                            matrices: vec![projection.clone()],
                            inputs: &[],
                            rows: 13,
                            columns: 64,
                            rows_per_batch: 0,
                            format: MatrixFormat::Packed(0),
                        },
                        &normalized,
                    )
                    .unwrap(),
            );
        }
        graph.end_projection_level().unwrap();
        let activated = graph.swiglu(&projected[0], &projected[1], None).unwrap();
        let repeated = graph.repeat(&activated, 2).unwrap();
        graph.begin_projection_level().unwrap();
        let down = graph.indexed(&bank, MatrixFormat::Packed(0), [3, 64, 13], &indices, 2, &repeated, true).unwrap();
        let shared = graph
            .mapped(
                &MappedOperation {
                    matrices: vec![shared.clone()],
                    inputs: &[],
                    rows: 64,
                    columns: 13,
                    rows_per_batch: 0,
                    format: MatrixFormat::Packed(0),
                },
                &activated,
            )
            .unwrap();
        graph.end_projection_level().unwrap();
        let mixed = graph.weighted_sum_device(&down, &probabilities, &shared, &shared_gate).unwrap();
        let result = graph.add(&normalized, &mixed).unwrap();
        let copy = graph.scratch(64).unwrap();
        graph.copy_into(&result, &copy).unwrap();
        let actual = graph.finish(&[result, copy]).unwrap();
        if let Some(expected) = &expected {
            assert!(actual == *expected, "single encoder changed dependent graph arithmetic");
        } else {
            expected = Some(actual);
        }
    }
}

#[stark_test]
fn single_encoder_retains_deferred_failure_checks_and_rejects_unsupported_transitions() {
    for separate_finite in [false, true] {
        let mut graph = DecodeGraph::new_parallel().unwrap();
        graph.serial_projection_levels = false;
        graph.separate_finite = separate_finite;
        let input = DecodeGraph::zeros(32).unwrap();
        unsafe { input.buffer.contents().as_ptr().cast::<f32>().write(f32::NAN) };
        graph.begin_projection_level().unwrap();
        let output = graph
            .mapped(
                &MappedOperation {
                    matrices: vec![test_matrix(&vec![0.0; 32])],
                    inputs: &[],
                    rows: 1,
                    columns: 32,
                    rows_per_batch: 0,
                    format: MatrixFormat::Packed(0),
                },
                &input,
            )
            .unwrap();
        graph.end_projection_level().unwrap();
        let _clipped = graph.swiglu(&output, &DecodeGraph::upload(&[1.0]).unwrap(), Some(1.0)).unwrap();
        assert!(graph.finish(&[]).is_err(), "single encoder lost a deferred failure");
    }
    let mut graph = DecodeGraph::new_parallel().unwrap();
    let input = DecodeGraph::upload(&[1.0, 2.0]).unwrap();
    let matrix = test_matrix(&[1.0]);
    assert!(
        graph
            .mapped(
                &MappedOperation {
                    matrices: vec![matrix.clone(), matrix],
                    inputs: &[],
                    rows: 1,
                    columns: 1,
                    rows_per_batch: 0,
                    format: MatrixFormat::Packed(0),
                },
                &input
            )
            .is_err(),
        "single encoder admitted an unsupported multi-matrix transition"
    );
    assert!(graph.switch_dispatch(MTLDispatchType::SERIAL).is_err());
    graph.serial_projection_levels = false;
    graph.begin_projection_level().unwrap();
    assert!(graph.finish(&[]).is_err(), "single encoder accepted an unfinished level");
}
