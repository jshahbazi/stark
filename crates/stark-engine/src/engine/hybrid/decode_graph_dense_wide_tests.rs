// SPDX-License-Identifier: Apache-2.0
//! Independent T8-kernel comparison, F64 decoded-weight products and ownership
//! gates for the opt-in true wide packed matrix operator.

use stark_engine_macros::stark_test;

use super::*;

#[stark_test]
fn dense_wide_geometry_bounds_storage_and_preserves_k_partitions() {
    for (rows, columns, nsg) in [(8192, 4096, 8), (4096, 8192, 16)] {
        for tokens in [32, 64, 128] {
            let geometry = wide_geometry(rows, columns, tokens).unwrap();
            assert_eq!(geometry.nsg, nsg);
            assert_eq!(geometry.rt * 8, tokens);
            assert_eq!(geometry.accumulator_elements(), 32);
            assert!(geometry.shared_bytes() <= 16 * 1024);
        }
    }
    for tokens in [0, 1, 8, 16, 31, 33, 65, 129, usize::MAX] {
        assert!(wide_geometry(8192, 4096, tokens).is_none());
    }
    for (rows, columns) in [(0, 4096), (8192, 0), (8191, 4096), (8192, 4095), (usize::MAX, 8192)] {
        assert!(wide_geometry(rows, columns, 64).is_none());
    }
}

fn finite_codes(
    kind: u32,
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let bytes = weights::block_layout(kind).unwrap().1;
    let mut state = 0x9135_acedu32 ^ kind;
    let mut codes = vec![0; rows * columns / 256 * bytes];
    for byte in &mut codes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    for (index, block) in codes.chunks_exact_mut(bytes).enumerate() {
        let d = half::f16::from_f32(0.001 + (index % 257) as f32 * 0.00003125).to_bits().to_le_bytes();
        if kind == 14 {
            block[208..210].copy_from_slice(&d);
        } else {
            block[..2].copy_from_slice(&d);
            block[2..4].copy_from_slice(&half::f16::from_f32(0.0004).to_bits().to_le_bytes());
        }
    }
    codes
}

fn operation(
    kind: u32,
    rows: usize,
    columns: usize,
    codes: &[u8],
) -> MappedOperation<'static> {
    let mut padded = vec![0x55; 7];
    padded.extend_from_slice(codes);
    padded.extend_from_slice(&[0xaa; 5]);
    MappedOperation {
        matrices: vec![MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), &padded).unwrap(),
                offset: 7,
                length: codes.len(),
                registered_length: padded.len(),
            },
            planes: None,
        }],
        inputs: &[],
        rows,
        columns,
        rows_per_batch: 0,
        format: MatrixFormat::Packed(kind),
    }
}

fn input_view(input: &[f32]) -> DeviceVector {
    let mut padded = vec![f32::NAN; 3];
    padded.extend_from_slice(input);
    padded.extend_from_slice(&[f32::NAN; 5]);
    DeviceVector {
        buffer: buffer(executor().unwrap(), bytemuck::cast_slice(&padded)).unwrap(),
        offset: 12,
        len: input.len(),
    }
}

/// Dispatch the unchanged T8 kernel directly for synthetic shapes. Its runtime
/// host wrapper intentionally rejects these small rows; its generic kernel is
/// the independent arithmetic reference, with the same nsg as the new tile.
fn t8_reference(
    operation: &MappedOperation<'_>,
    input: &DeviceVector,
    tokens: usize,
    geometry: WideGeometry,
) -> Vec<f32> {
    let MatrixFormat::Packed(kind) = operation.format else {
        unreachable!()
    };
    let pipeline = super::super::pipeline(kind, geometry.nsg).unwrap();
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let output = graph.scratch(tokens * operation.rows).unwrap();
    let matrix = &operation.matrices[0];
    graph.resources.extend([matrix.codes.buffer.clone(), input.buffer.clone(), output.buffer.clone()]);
    graph.check_finite(input).unwrap();
    for start in (0..tokens).step_by(8) {
        graph.order_dispatch();
        graph.encoding.set_compute_pipeline_state(&pipeline);
        for (index, buffer) in [&matrix.codes.buffer, &input.buffer, &output.buffer].into_iter().enumerate() {
            graph.encoding.set_buffer(Some(buffer), 0, index);
        }
        let p = [
            operation.rows as u32,
            operation.columns as u32,
            (tokens - start).min(8) as u32,
            matrix.codes.offset as u32,
            (input.offset / 4 + start * operation.columns) as u32,
            (output.offset / 4 + start * operation.rows) as u32,
        ];
        graph.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        graph.encoding.set_buffer(Some(&graph.failure.buffer), graph.failure.offset, 4);
        graph.encoding.set_threadgroup_memory_length(geometry.nsg * 4 * 256, 0);
        graph.encoding.dispatch_threadgroups(
            MTLSize {
                width: operation.rows.div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: geometry.nsg * 32,
                height: 1,
                depth: 1,
            },
        );
    }
    graph.finish(&[output]).unwrap().remove(0)
}

fn run_wide(
    operation: &MappedOperation<'_>,
    input: &DeviceVector,
    tokens: usize,
    kind: u32,
    geometry: WideGeometry,
) -> Result<Vec<f32>> {
    let mut graph = DecodeGraph::new_unretained()?;
    let arena = graph.upload_into(&vec![73.25; tokens * operation.rows + 23])?;
    let output = arena.slice(17, tokens * operation.rows)?;
    if !graph.encode_dense_wide_geometry(operation, input, &output, tokens, kind, geometry)? {
        return Err(Error::Unsupported("test dense wide pipeline was not admitted"));
    }
    let mut values = graph.finish(&[output, arena])?;
    let arena = values.pop().unwrap();
    assert!(
        arena[..17]
            .iter()
            .chain(&arena[tokens * operation.rows + 17..])
            .all(|value| value.to_bits() == 73.25f32.to_bits()),
        "dense wide wrote beyond its output view"
    );
    Ok(values.remove(0))
}

fn check_f64(
    kind: u32,
    rows: usize,
    columns: usize,
    codes: &[u8],
    input: &[f32],
    output: &[f32],
) {
    let block_bytes = weights::block_layout(kind).unwrap().1;
    let row_bytes = columns / 256 * block_bytes;
    for (token, values) in input.chunks_exact(columns).enumerate() {
        for row in 0..rows {
            let mut total = 0.0f64;
            let mut magnitude = 0.0f64;
            for (block, activation) in
                codes[row * row_bytes..(row + 1) * row_bytes].chunks_exact(block_bytes).zip(values.chunks_exact(256))
            {
                let mut decoded = [0.0f32; 256];
                weights::decode_block(kind, block, &mut decoded).unwrap();
                for (weight, value) in decoded.into_iter().zip(activation) {
                    let product = f64::from(weight) * f64::from(*value);
                    total += product;
                    magnitude += product.abs();
                }
            }
            assert!(
                (f64::from(output[token * rows + row]) - total).abs() / magnitude.max(1e-30) <= 2e-6,
                "dense wide decoded-F64 normalized error exceeded gate"
            );
        }
    }
}

#[stark_test]
#[ignore]
fn dense_wide_rejects_invalid_views_and_descriptors_before_binding() {
    let (rows, columns, tokens, kind) = (35, 1024, 32, 14);
    let geometry = WideGeometry {
        nsg: 8,
        nt: 4,
        rt: 4,
    };
    let mut operation = operation(kind, rows, columns, &finite_codes(kind, rows, columns));
    let input = input_view(&vec![0.25; tokens * columns]);
    let mut graph = DecodeGraph::new_unretained().unwrap();
    let output = graph.scratch(tokens * rows).unwrap();
    let resources = graph.resources.len();
    assert!(
        graph
            .encode_dense_wide_geometry(
                &operation,
                &input.slice(0, input.len - 1).unwrap(),
                &output,
                tokens,
                kind,
                geometry
            )
            .is_err()
    );
    assert!(
        graph
            .encode_dense_wide_geometry(
                &operation,
                &input,
                &output.slice(0, output.len - 1).unwrap(),
                tokens,
                kind,
                geometry
            )
            .is_err()
    );
    let alias = input.slice(0, output.len).unwrap();
    assert!(graph.encode_dense_wide_geometry(&operation, &input, &alias, tokens, kind, geometry).is_err());
    operation.matrices[0].codes.length += 1;
    assert!(graph.encode_dense_wide_geometry(&operation, &input, &output, tokens, kind, geometry).is_err());
    operation.matrices[0].codes.length -= 1;
    let mapped_alias = DeviceVector {
        buffer: operation.matrices[0].codes.buffer.clone(),
        offset: 8,
        len: output.len,
    };
    assert!(graph.encode_dense_wide_geometry(&operation, &input, &mapped_alias, tokens, kind, geometry).is_err());
    assert_eq!(graph.resources.len(), resources, "invalid descriptor changed bound resource ownership");
}

#[stark_test]
#[ignore]
fn dense_wide_matches_t8_kernel_f64_tails_and_offset_spans() {
    let (rows, columns) = (35, 1024);
    for kind in [13, 14] {
        let codes = finite_codes(kind, rows, columns);
        let operation = operation(kind, rows, columns, &codes);
        for maximum in [32, 64, 128] {
            for nsg in [8, 16] {
                let geometry = WideGeometry {
                    nsg,
                    nt: 128 / maximum,
                    rt: maximum / 8,
                };
                for tokens in [maximum - 1, maximum] {
                    let input: Vec<_> = (0..tokens * columns).map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.013).collect();
                    let view = input_view(&input);
                    let expected = t8_reference(&operation, &view, tokens, geometry);
                    let actual = run_wide(&operation, &view, tokens, kind, geometry).unwrap();
                    assert!(
                        actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "wide operator changed a T8 output's F32 arithmetic"
                    );
                    check_f64(kind, rows, columns, &codes, &input, &actual);
                }
            }
        }
    }
}

#[stark_test]
#[ignore]
fn dense_wide_extreme_fallback_is_scoped_to_each_t8_and_nonfinite_is_sticky() {
    let (rows, columns, tokens) = (35, 1024, 128);
    let geometry = WideGeometry {
        nsg: 8,
        nt: 1,
        rt: 16,
    };
    for kind in [13, 14] {
        let codes = finite_codes(kind, rows, columns);
        let operation = operation(kind, rows, columns, &codes);
        let mut input: Vec<_> = (0..tokens * columns).map(|i| ((i * 13 % 71) as f32 - 35.0) * 0.017).collect();
        input[8 * columns + 255] = 1e21;
        let view = input_view(&input);
        let expected = t8_reference(&operation, &view, tokens, geometry);
        let actual = run_wide(&operation, &view, tokens, kind, geometry).unwrap();
        assert!(
            actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
            "an extreme T8 group changed another group's reduction tree"
        );
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            input[(tokens - 1) * columns + 255] = poison;
            assert!(
                run_wide(&operation, &input_view(&input), tokens, kind, geometry).is_err(),
                "dense wide concealed a nonfinite input"
            );
        }
    }
}
