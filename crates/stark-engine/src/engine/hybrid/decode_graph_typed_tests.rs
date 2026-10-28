// SPDX-License-Identifier: Apache-2.0
//! Aligned affine-load admission and exact equivalence to the byte decoder.

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
fn aligned_affine_graph_loads_match_bytes_for_all_formats_and_unaligned_fallbacks() {
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
                for offsets in [[64, 128, 192], [65, 128, 192], [64, 129, 192], [64, 128, 193], [64, 130, 194]] {
                    let matrix = MappedMatrix {
                        codes: span(&codes, offsets[0]),
                        planes: Some((span(&encode(&scales), offsets[1]), span(&encode(&biases), offsets[2]))),
                    };
                    let alignment = if scale_dtype == 0 {
                        4
                    } else {
                        2
                    };
                    assert_eq!(
                        matrix.aligned_affine_loads(format),
                        offsets[0] % 4 == 0 && offsets[1] % alignment == 0 && offsets[2] % alignment == 0
                    );
                    assert!(!matrix.aligned_affine_loads(MatrixFormat::Packed(0)));
                    for batched in [false, true] {
                        let width = columns
                            * if batched {
                                experts
                            } else {
                                1
                            };
                        let input = (0..width).map(|i| ((i % 23) as f32 - 11.0) * 0.043).collect::<Vec<_>>();
                        let mut expected = None;
                        for byte_loads in [true, false] {
                            let mut graph = DecodeGraph::new().unwrap();
                            graph.byte_loads = byte_loads;
                            graph.vector8 = true;
                            let input = graph.upload_into(&input).unwrap();
                            let output = graph
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
                                    &input,
                                )
                                .unwrap();
                            let actual = graph
                                .finish(&[output])
                                .unwrap()
                                .remove(0)
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>();
                            if let Some(expected) = &expected {
                                assert_eq!(&actual, expected, "typed mapped affine loads changed F32 bits");
                            } else {
                                expected = Some(actual);
                            }
                        }

                        let mut expected = None;
                        for byte_loads in [true, false] {
                            let mut graph = DecodeGraph::new().unwrap();
                            graph.byte_loads = byte_loads;
                            graph.vector8 = true;
                            let input = graph.upload_into(&input).unwrap();
                            let ids = graph.upload_into(&[f32::from_bits(2), 0.0, f32::from_bits(2)]).unwrap();
                            let output = graph
                                .indexed(&matrix, format, [experts, rows, columns], &ids, 3, &input, batched)
                                .unwrap();
                            let actual = graph
                                .finish(&[output])
                                .unwrap()
                                .remove(0)
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>();
                            if let Some(expected) = &expected {
                                assert_eq!(&actual, expected, "typed indexed affine loads changed F32 bits");
                            } else {
                                expected = Some(actual);
                            }
                        }
                    }
                }
            }
        }
    }
}
