// SPDX-License-Identifier: Apache-2.0
//! Qualify the 32-value lane schedule against the vector8 affine baseline.

use stark_engine_macros::stark_test;

use super::*;

fn span(
    bytes: &[u8],
    offset: usize,
) -> MappedSlice {
    let mut storage = vec![0u8; offset];
    storage.extend_from_slice(bytes);
    MappedSlice {
        buffer: buffer(executor().unwrap(), &storage).unwrap(),
        offset,
        length: bytes.len(),
        registered_length: storage.len(),
    }
}

#[stark_test]
fn affine_uint4_chunk_loads_preserve_scalar_uint_bits_and_alignment_fallbacks() {
    for bits in [4usize, 8] {
        for group in [32usize, 64, 128] {
            for scale_dtype in [0, 1, 30] {
                let (experts, rows, columns) = (3, 13, group * 3);
                let codes = (0..experts * rows * columns * bits / 8).map(|i| (i * 19 + 37) as u8).collect::<Vec<_>>();
                let scales =
                    (0..experts * rows * columns / group).map(|i| ((i % 13 + 1) as f32) * 0.013).collect::<Vec<_>>();
                let biases = scales.iter().enumerate().map(|(i, _)| -((i % 7) as f32) * 0.071).collect::<Vec<_>>();
                let encode = |values: &[f32]| {
                    values
                        .iter()
                        .flat_map(|&value| match scale_dtype {
                            0 => value.to_le_bytes().to_vec(),
                            1 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                            _ => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                        })
                        .collect::<Vec<_>>()
                };
                let format = MatrixFormat::Affine {
                    scale_dtype,
                    bits,
                    group,
                };
                for offset in [64usize, 68, 65] {
                    let matrix = MappedMatrix {
                        codes: span(&codes, offset),
                        planes: Some((span(&encode(&scales), 128), span(&encode(&biases), 192))),
                    };
                    for batched in [false, true] {
                        let input = (0..columns
                            * if batched {
                                experts
                            } else {
                                1
                            })
                            .map(|i| ((i % 23) as f32 - 11.0) * 0.043)
                            .collect::<Vec<_>>();
                        for indexed in [false, true] {
                            let mut reference = None;
                            for scalar_uint_loads in [true, false] {
                                let mut graph = DecodeGraph::new().unwrap();
                                graph.vector8 = false;
                                graph.scalar_uint_loads = scalar_uint_loads;
                                graph.byte_loads = false;
                                graph.separate_finite = false;
                                let x = graph.upload_into(&input).unwrap();
                                let output = if indexed {
                                    let ids = graph.upload_into(&[f32::from_bits(2), 0.0, f32::from_bits(2)]).unwrap();
                                    graph
                                        .indexed(&matrix, format, [experts, rows, columns], &ids, 3, &x, batched)
                                        .unwrap()
                                } else {
                                    graph
                                        .mapped(
                                            &MappedOperation {
                                                matrices: vec![matrix.clone()],
                                                inputs: &[],
                                                rows: experts * rows,
                                                columns,
                                                rows_per_batch: if batched {
                                                    rows
                                                } else {
                                                    0
                                                },
                                                format,
                                            },
                                            &x,
                                        )
                                        .unwrap()
                                };
                                let actual = graph
                                    .finish(&[output])
                                    .unwrap()
                                    .remove(0)
                                    .into_iter()
                                    .map(f32::to_bits)
                                    .collect::<Vec<_>>();
                                if let Some(reference) = &reference {
                                    assert_eq!(&actual, reference, "uint4 loads changed affine chunk32 arithmetic");
                                } else {
                                    reference = Some(actual);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[stark_test]
fn affine_chunk32_matches_vector8_across_groups_dtypes_shapes_and_fallbacks() {
    for bits in [4usize, 8] {
        for group in [32usize, 64, 128] {
            for scale_dtype in [0, 1, 30] {
                for columns in [32, 64, 96, 128, 320, 640, 1024, 1056, 2560, 10240] {
                    if columns % group != 0 {
                        continue;
                    }
                    let (experts, rows) = (3, 13);
                    let codes =
                        (0..experts * rows * columns * bits / 8).map(|i| (i * 19 + 37) as u8).collect::<Vec<_>>();
                    let scales = (0..experts * rows * columns / group)
                        .map(|i| ((i % 13 + 1) as f32) * 0.0003)
                        .collect::<Vec<_>>();
                    let biases = scales.iter().enumerate().map(|(i, _)| -((i % 7) as f32) * 0.0009).collect::<Vec<_>>();
                    let encode = |values: &[f32]| {
                        values
                            .iter()
                            .flat_map(|&value| match scale_dtype {
                                0 => value.to_le_bytes().to_vec(),
                                1 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                                _ => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                            })
                            .collect::<Vec<_>>()
                    };
                    let format = MatrixFormat::Affine {
                        scale_dtype,
                        bits,
                        group,
                    };
                    for offset in [64usize, 65] {
                        let matrix = MappedMatrix {
                            codes: span(&codes, offset),
                            planes: Some((span(&encode(&scales), 128), span(&encode(&biases), 192))),
                        };
                        for batched in [false, true] {
                            let width = columns
                                * if batched {
                                    experts
                                } else {
                                    1
                                };
                            let input = (0..width).map(|i| ((i % 23) as f32 - 11.0) * 0.004).collect::<Vec<_>>();
                            for indexed in [false, true] {
                                let mut reference = None;
                                for vector8 in [true, false] {
                                    let mut graph = DecodeGraph::new().unwrap();
                                    graph.vector8 = vector8;
                                    graph.byte_loads = false;
                                    graph.separate_finite = false;
                                    let x = graph.upload_into(&input).unwrap();
                                    let output = if indexed {
                                        let ids =
                                            graph.upload_into(&[f32::from_bits(2), 0.0, f32::from_bits(2)]).unwrap();
                                        graph
                                            .indexed(&matrix, format, [experts, rows, columns], &ids, 3, &x, batched)
                                            .unwrap()
                                    } else {
                                        graph
                                            .mapped(
                                                &MappedOperation {
                                                    matrices: vec![matrix.clone()],
                                                    inputs: &[],
                                                    rows: experts * rows,
                                                    columns,
                                                    rows_per_batch: if batched {
                                                        rows
                                                    } else {
                                                        0
                                                    },
                                                    format,
                                                },
                                                &x,
                                            )
                                            .unwrap()
                                    };
                                    let actual = graph.finish(&[output]).unwrap().remove(0);
                                    if let Some(reference) = &reference {
                                        let reference: &Vec<f32> = reference;
                                        for (&actual, &expected) in actual.iter().zip(reference) {
                                            if offset == 65 {
                                                assert_eq!(
                                                    actual.to_bits(),
                                                    expected.to_bits(),
                                                    "unaligned chunk32 fallback changed arithmetic"
                                                );
                                            } else {
                                                assert!(
                                                    (actual - expected).abs() <= 2e-5,
                                                    "chunk32 affine reduction exceeded source tolerance"
                                                );
                                            }
                                        }
                                    } else {
                                        reference = Some(actual);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
