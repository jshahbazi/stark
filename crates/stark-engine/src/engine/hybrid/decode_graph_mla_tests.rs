// SPDX-License-Identifier: Apache-2.0
//! Independent old-kernel and decoded-F64 gates for token projection reuse.

use stark_engine_macros::stark_test;

use super::*;

fn encoded(
    kind: u32,
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let (block, bytes) = weights::block_layout(kind).unwrap();
    let mut state = 0x7193_43c5u32 ^ kind;
    let mut result = vec![0; rows * columns / block * bytes];
    for value in &mut result {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *value = (state >> 24) as u8;
    }
    for (index, chunk) in result.chunks_exact_mut(bytes).enumerate() {
        let value = 0.001 + (index % 127) as f32 * 0.00003125;
        match kind {
            0 => chunk.copy_from_slice(&((index as f32 * 0.17).sin() * 0.1).to_le_bytes()),
            1 => chunk.copy_from_slice(&half::f16::from_f32((index as f32 * 0.17).sin() * 0.1).to_bits().to_le_bytes()),
            30 => {
                chunk.copy_from_slice(&half::bf16::from_f32((index as f32 * 0.17).sin() * 0.1).to_bits().to_le_bytes())
            },
            8 => chunk[..2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes()),
            13 => {
                chunk[..2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
                chunk[2..4].copy_from_slice(&half::f16::from_f32(0.0004).to_bits().to_le_bytes());
            },
            14 => chunk[208..210].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes()),
            _ => unreachable!(),
        }
    }
    result
}

fn operation(
    kind: u32,
    rows: usize,
    columns: usize,
    group: usize,
    bytes: &[u8],
) -> MappedOperation<'static> {
    let mut padded = vec![0x55; 7];
    padded.extend_from_slice(bytes);
    padded.extend_from_slice(&[0xaa; 5]);
    MappedOperation {
        matrices: vec![MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), &padded).unwrap(),
                offset: 7,
                length: bytes.len(),
                registered_length: padded.len(),
            },
            planes: None,
        }],
        inputs: &[],
        rows,
        columns,
        rows_per_batch: group,
        format: MatrixFormat::Packed(kind),
    }
}

fn input_view(values: &[f32]) -> DeviceVector {
    let mut padded = vec![f32::NAN; 3];
    padded.extend_from_slice(values);
    padded.extend_from_slice(&[f32::NAN; 5]);
    DeviceVector {
        buffer: buffer(executor().unwrap(), bytemuck::cast_slice(&padded)).unwrap(),
        offset: 12,
        len: values.len(),
    }
}

fn old_reference(
    op: &MappedOperation<'_>,
    input: &DeviceVector,
    tokens: usize,
    schedule: TokenSchedule,
) -> Result<Vec<f32>> {
    let (name, rows_per_group, threads) = match schedule {
        TokenSchedule::Generic => ("hybrid_packed_matvec", 1, 256),
        TokenSchedule::Q8Grouped => ("hybrid_q8_grouped8_r4g2_matvec", 4, 64),
        TokenSchedule::Q5Grouped => ("hybrid_q5_grouped32_matvec", 2, 64),
        TokenSchedule::Q6Grouped => ("hybrid_q6_grouped16_matvec", 4, 64),
    };
    let pipeline = executor()?.cached_pipeline(op.format, name, &Mutex::new(HashMap::new()))?;
    let mut graph = DecodeGraph::new_unretained()?;
    let output = graph.scratch(tokens * op.rows)?;
    let stride = input.len / tokens;
    let parameters = op.parameters_for_input(stride)?;
    graph.check_finite(input)?;
    for token in 0..tokens {
        graph.order_dispatch();
        graph.encoding.set_compute_pipeline_state(&pipeline);
        let input_at = input.offset + token * stride * 4;
        graph.encoding.set_buffer(Some(&op.matrices[0].codes.buffer), 0, 0);
        graph.encoding.set_buffer(Some(&input.buffer), input_at, 1);
        graph.encoding.set_buffer(Some(&output.buffer), output.offset + token * op.rows * 4, 2);
        graph.encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
        let offsets = [op.matrices[0].codes.offset as u32, 0, 0, 0];
        graph.encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
        graph.encoding.dispatch_threadgroups(
            MTLSize {
                width: op.rows.div_ceil(rows_per_group),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }
    graph.resources.extend([op.matrices[0].codes.buffer.clone(), input.buffer.clone(), output.buffer.clone()]);
    graph.check_finite(&output)?;
    Ok(graph.finish(&[output])?.remove(0))
}

fn candidate(
    op: &MappedOperation<'_>,
    input: &DeviceVector,
    tokens: usize,
    schedule: TokenSchedule,
) -> Result<Vec<f32>> {
    let mut graph = DecodeGraph::new_unretained()?;
    let count = tokens * op.rows;
    let padded = vec![7.25_f32; count + 8];
    let storage = buffer(executor()?, bytemuck::cast_slice(&padded))?;
    let output = DeviceVector {
        buffer: storage.clone(),
        offset: 12,
        len: count,
    };
    if !graph.encode_token_projection(op, input, &output, tokens, schedule)? {
        return Err(Error::Unsupported("test token projection pipeline unavailable"));
    }
    let values = graph.finish(&[output])?.remove(0);
    let guarded = unsafe { std::slice::from_raw_parts(storage.contents().as_ptr().cast::<f32>(), padded.len()) };
    assert!(
        guarded[..3].iter().chain(&guarded[3 + count..]).all(|&value| value == 7.25),
        "token projection guards changed"
    );
    Ok(values)
}

fn check_f64(
    kind: u32,
    codes: &[u8],
    op: &MappedOperation<'_>,
    input: &[f32],
    actual: &[f32],
    tokens: usize,
) {
    let (block, bytes) = weights::block_layout(kind).unwrap();
    let row_bytes = op.columns / block * bytes;
    let stride = input.len() / tokens;
    for token in 0..tokens {
        for row in 0..op.rows {
            let start = token * stride
                + if op.rows_per_batch == 0 {
                    0
                } else {
                    row / op.rows_per_batch * op.columns
                };
            let mut expected = 0.0_f64;
            let mut absolute = 0.0_f64;
            for b in 0..op.columns / block {
                let mut decoded = vec![0.0; block];
                let at = row * row_bytes + b * bytes;
                weights::decode_block(kind, &codes[at..at + bytes], &mut decoded).unwrap();
                for (i, weight) in decoded.into_iter().enumerate() {
                    let product = f64::from(weight) * f64::from(input[start + b * block + i]);
                    expected += product;
                    absolute += product.abs();
                }
            }
            assert!(
                (f64::from(actual[token * op.rows + row]) - expected).abs() <= 2e-5 * absolute.max(1.0),
                "token projection decoded F64 gate"
            );
        }
    }
}

fn exact(
    a: &[f32],
    b: &[f32],
) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[stark_test]
fn mla_token_projection_generic_preserves_heads_offsets_and_tails() {
    for kind in [0, 1, 30, 8] {
        let columns = if kind == 8 {
            64
        } else {
            37
        };
        for group in [0, 5] {
            let rows = if group == 0 {
                13
            } else {
                15
            };
            let codes = encoded(kind, rows, columns);
            let op = operation(kind, rows, columns, group, &codes);
            for tokens in [1, 3, 9, 17] {
                let stride = columns
                    * if group == 0 {
                        1
                    } else {
                        rows / group
                    };
                let values: Vec<_> = (0..tokens * stride).map(|i| (i as f32 * 0.019).sin() * 0.25).collect();
                let input = input_view(&values);
                let expected = old_reference(&op, &input, tokens, TokenSchedule::Generic).unwrap();
                let actual = candidate(&op, &input, tokens, TokenSchedule::Generic).unwrap();
                assert!(exact(&actual, &expected), "generic token projection changed row bits");
                check_f64(kind, &codes, &op, &values, &actual, tokens);
            }
        }
    }
}

#[stark_test]
fn mla_token_projection_grouped_policies_preserve_bits_and_extreme_scope() {
    for (kind, schedule, columns) in
        [(8, TokenSchedule::Q8Grouped, 1536), (13, TokenSchedule::Q5Grouped, 512), (14, TokenSchedule::Q6Grouped, 512)]
    {
        let codes = encoded(kind, 13, columns);
        let op = operation(kind, 13, columns, 0, &codes);
        for tokens in [1, 3, 5, 9] {
            for extreme in [false, true] {
                let mut values: Vec<_> = (0..tokens * columns).map(|i| (i as f32 * 0.017).sin() * 0.25).collect();
                if extreme {
                    for value in &mut values[(tokens - 1) * columns..] {
                        *value *= 1e30;
                    }
                }
                let input = input_view(&values);
                let expected = old_reference(&op, &input, tokens, schedule).unwrap();
                let actual = candidate(&op, &input, tokens, schedule).unwrap();
                assert!(exact(&actual, &expected), "grouped token projection changed row/fallback bits");
                if !extreme {
                    check_f64(kind, &codes, &op, &values, &actual, tokens);
                }
            }
        }
    }
    // The selected Q8 schedule falls back for an output tile spanning heads.
    let codes = encoded(8, 39, 1536);
    let op = operation(8, 39, 1536, 13, &codes);
    let input = input_view(&(0..9 * 3 * 1536).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>());
    assert!(
        exact(
            &candidate(&op, &input, 9, TokenSchedule::Q8Grouped).unwrap(),
            &old_reference(&op, &input, 9, TokenSchedule::Q8Grouped).unwrap()
        ),
        "Q8 cross-head fallback changed"
    );
}

#[stark_test]
fn mla_token_projection_rejects_invalid_spans_aliases_and_nonfinite_inputs() {
    let op = operation(8, 13, 64, 0, &encoded(8, 13, 64));
    let input = input_view(&vec![0.25; 3 * 64]);
    for tokens in [0, 2, 129] {
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let output = graph.scratch(3 * 13).unwrap();
        assert!(graph.encode_token_projection(&op, &input, &output, tokens, TokenSchedule::Generic).is_err());
    }
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let aliased = input.slice(0, 3 * 13).unwrap();
    assert!(graph.encode_token_projection(&op, &input, &aliased, 3, TokenSchedule::Generic).is_err());
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let invalid = input_view(&vec![0.0; 3 * 64]);
        unsafe {
            invalid.buffer.contents().as_ptr().cast::<u8>().add(invalid.offset).cast::<f32>().add(2 * 64).write(poison);
        }
        assert!(
            candidate(&op, &invalid, 3, TokenSchedule::Generic).is_err(),
            "nonfinite token projection input escaped"
        );
    }
    assert!(candidate(&op, &input, 3, TokenSchedule::Generic).is_ok());
    for malformed in [0, 1, 2] {
        let mut invalid = operation(8, 13, 64, 0, &encoded(8, 13, 64));
        let matrix = &mut invalid.matrices[0];
        match malformed {
            0 => matrix.codes.length = matrix.codes.buffer.length(),
            1 => matrix.codes.offset = matrix.codes.buffer.length() - 1,
            _ => matrix.planes = Some((matrix.codes.clone(), matrix.codes.clone())),
        }
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let output = graph.scratch(3 * 13).unwrap();
        assert!(
            graph.encode_token_projection(&invalid, &input, &output, 3, TokenSchedule::Generic).is_err(),
            "invalid weight span accepted"
        );
    }
    let aliased = DeviceVector {
        buffer: op.matrices[0].codes.buffer.clone(),
        offset: 8,
        len: 3 * 13,
    };
    let mut graph = DecodeGraph::new_unretained().unwrap();
    assert!(
        graph.encode_token_projection(&op, &input, &aliased, 3, TokenSchedule::Generic).is_err(),
        "immutable weight output alias accepted"
    );
}

#[stark_test]
fn mla_token_projection_queued_graph_retains_source_and_activation_owners() {
    let (pending, expected) = {
        let op = operation(8, 13, 64, 0, &encoded(8, 13, 64));
        let input = input_view(&(0..9 * 64).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>());
        let expected = old_reference(&op, &input, 9, TokenSchedule::Generic).unwrap();
        let mut graph = DecodeGraph::new_unretained().unwrap();
        let output = graph.scratch(9 * 13).unwrap();
        assert!(graph.encode_token_projection(&op, &input, &output, 9, TokenSchedule::Generic).unwrap());
        (graph.submit(&[output]).unwrap(), expected)
    };
    assert!(exact(&pending.wait().unwrap().remove(0), &expected), "queued token projection lost an owner");
}

#[stark_test]
fn mla_batch_index_norm_matches_original_rows() {
    for width in [3, 128] {
        for tokens in [1, 3, 17, 128] {
            let input = input_view(&(0..width * tokens).map(|i| (i as f32 * 0.023).sin()).collect::<Vec<_>>());
            let weight = DecodeGraph::upload(&vec![0.9; width]).unwrap();
            let bias = DecodeGraph::upload(&vec![0.03; width]).unwrap();
            let mut old = DecodeGraph::new_unretained().unwrap();
            let mut rows = Vec::new();
            for token in 0..tokens {
                rows.push(
                    crate::engine::hybrid::glm_metal::index_layer_norm(
                        &mut old,
                        &input.slice(token * width, width).unwrap(),
                        &weight,
                        &bias,
                    )
                    .unwrap(),
                );
            }
            let expected: Vec<_> = old.finish(&rows).unwrap().into_iter().flatten().collect();
            let mut graph = DecodeGraph::new_unretained().unwrap();
            let output = graph.mla_index_norm_batch(&input, &weight, &bias, width, tokens).unwrap();
            let actual = graph.finish(&[output]).unwrap().remove(0);
            assert!(exact(&actual, &expected), "batched index normalization changed row bits");
        }
    }
}
