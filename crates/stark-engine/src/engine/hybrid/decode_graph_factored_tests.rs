// SPDX-License-Identifier: Apache-2.0
//! Independent F64 and unchanged-product checks for the optional affine dot.

use stark_engine_macros::stark_test;

use super::*;

fn plane(
    values: &[f32],
    kind: u32,
) -> (MappedSlice, Vec<f32>) {
    let rounded = values
        .iter()
        .map(|&value| match kind {
            0 => value,
            1 => half::f16::from_f32(value).to_f32(),
            _ => half::bf16::from_f32(value).to_f32(),
        })
        .collect::<Vec<_>>();
    let bytes = rounded
        .iter()
        .flat_map(|&value| match kind {
            0 => value.to_le_bytes().to_vec(),
            1 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
            _ => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
        })
        .collect::<Vec<_>>();
    (
        MappedSlice {
            buffer: buffer(executor().unwrap(), &bytes).unwrap(),
            offset: 0,
            length: bytes.len(),
            registered_length: bytes.len(),
        },
        rounded,
    )
}

fn make_matrix(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    format: MatrixFormat,
) -> (MappedMatrix, Vec<f32>) {
    let MatrixFormat::Affine {
        bits,
        group,
        scale_dtype,
    } = format
    else {
        unreachable!()
    };
    let packed = if bits == 4 {
        codes.chunks_exact(2).map(|pair| pair[0] | pair[1] << 4).collect::<Vec<_>>()
    } else {
        codes.to_vec()
    };
    let (scales, scale_values) = plane(scales, scale_dtype);
    let (biases, bias_values) = plane(biases, scale_dtype);
    let weights = codes
        .iter()
        .enumerate()
        .map(|(i, &q)| scale_values[i / group] * f32::from(q) + bias_values[i / group])
        .collect();
    (
        MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), &packed).unwrap(),
                offset: 0,
                length: packed.len(),
                registered_length: packed.len(),
            },
            planes: Some((scales, biases)),
        },
        weights,
    )
}

fn project(
    matrix: &MappedMatrix,
    format: MatrixFormat,
    shape: [usize; 3],
    input: &[f32],
    indexed: bool,
    batched: bool,
    factored: bool,
) -> Result<Vec<f32>> {
    let [experts, rows, columns] = shape;
    let mut graph = DecodeGraph::new()?;
    graph.vector8 = false;
    graph.scalar_uint_loads = false;
    graph.byte_loads = false;
    graph.separate_finite = false;
    graph.factored_affine = factored;
    let x = DecodeGraph::zeros(input.len())?;
    // Test storage deliberately supports poison values rejected by host upload.
    unsafe { x.buffer.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(input.as_ptr(), input.len()) };
    let output = if indexed {
        let ids = graph.upload_into(&[f32::from_bits(2), 0.0, f32::from_bits(2)])?;
        graph.indexed(matrix, format, shape, &ids, 3, &x, batched)?
    } else {
        graph.mapped(
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
        )?
    };
    graph.finish(&[output]).map(|mut outputs| outputs.remove(0))
}

#[stark_test]
fn factored_affine_matches_f64_and_product_baseline_across_all_layouts() {
    let (experts, rows) = (3, 13);
    for bits in [4usize, 8] {
        for group in [32usize, 64, 128] {
            for scale_dtype in [0, 1, 30] {
                for columns in [32, 64, 96, 128, 320, 640, 1024, 1056, 2560] {
                    if columns % group != 0 {
                        continue;
                    }
                    let count = experts * rows * columns;
                    let codes = (0..count).map(|i| ((i * 19 + 37) & ((1 << bits) - 1)) as u8).collect::<Vec<_>>();
                    let scales = (0..count / group).map(|i| (i % 13 + 1) as f32 * 0.0003).collect::<Vec<_>>();
                    let biases = (0..count / group).map(|i| -((i % 7) as f32) * 0.0007).collect::<Vec<_>>();
                    let format = MatrixFormat::Affine {
                        bits,
                        group,
                        scale_dtype,
                    };
                    let (matrix, weights) = make_matrix(&codes, &scales, &biases, format);
                    for batched in [false, true] {
                        let input = (0..columns
                            * if batched {
                                3
                            } else {
                                1
                            })
                            .map(|i| (i % 23) as f32 * 0.004 - 0.044)
                            .collect::<Vec<_>>();
                        for indexed in [false, true] {
                            let baseline =
                                project(&matrix, format, [experts, rows, columns], &input, indexed, batched, false)
                                    .unwrap();
                            let actual =
                                project(&matrix, format, [experts, rows, columns], &input, indexed, batched, true)
                                    .unwrap();
                            for (out, (&actual, &baseline)) in actual.iter().zip(&baseline).enumerate() {
                                let slot = out / rows;
                                let expert = if indexed {
                                    [2, 0, 2][slot]
                                } else {
                                    slot
                                };
                                let weight_base = (expert * rows + out % rows) * columns;
                                let input_base = if batched {
                                    slot * columns
                                } else {
                                    0
                                };
                                let oracle = (0..columns)
                                    .map(|i| f64::from(weights[weight_base + i]) * f64::from(input[input_base + i]))
                                    .sum::<f64>();
                                assert!(
                                    (actual - baseline).abs() <= 2e-5,
                                    "factored affine exceeded product baseline tolerance"
                                );
                                assert!(
                                    (f64::from(actual) - oracle).abs() <= 2e-5,
                                    "factored affine exceeded independent F64 tolerance"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[stark_test]
fn factored_affine_preserves_extreme_zero_cancellation_and_poison_contracts() {
    let shape = [3, 13, 1024];
    let count = shape.iter().product::<usize>();
    let format = MatrixFormat::Affine {
        bits: 8,
        group: 32,
        scale_dtype: 0,
    };
    for (scale, bias, input_value) in
        [(0.0, 0.0, f32::MAX), (1.0, -127.0, f32::MAX), (1.0, -127.0, 1e30), (0.0, 0.25, 1.0)]
    {
        let (matrix, _) = make_matrix(&vec![127; count], &vec![scale; count / 32], &vec![bias; count / 32], format);
        let input = vec![input_value; shape[2]];
        for indexed in [false, true] {
            let baseline = project(&matrix, format, shape, &input, indexed, false, false).unwrap();
            let actual = project(&matrix, format, shape, &input, indexed, false, true).unwrap();
            assert_eq!(actual, baseline, "factored affine changed valid extreme cancellation");
        }
    }
    let (matrix, _) = make_matrix(&vec![127; count], &vec![1.0; count / 32], &vec![-127.0; count / 32], format);
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for indexed in [false, true] {
            assert!(project(&matrix, format, shape, &vec![poison; shape[2]], indexed, false, true).is_err());
        }
    }
    let codes = (0..count)
        .map(|i| {
            if i % 2 == 0 {
                126
            } else {
                128
            }
        })
        .collect::<Vec<_>>();
    let (matrix, _) = make_matrix(&codes, &vec![1.0; count / 32], &vec![-127.0; count / 32], format);
    let input = (0..shape[2]).map(|i| (i % 5) as f32 * 4096.0 - 8192.0).collect::<Vec<_>>();
    let baseline = project(&matrix, format, shape, &input, false, false, false).unwrap();
    let actual = project(&matrix, format, shape, &input, false, false, true).unwrap();
    assert_eq!(actual, baseline, "factored Q8 cancellation changed exact integer result");
}
