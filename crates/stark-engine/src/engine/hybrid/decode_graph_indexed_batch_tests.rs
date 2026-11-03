// SPDX-License-Identifier: Apache-2.0

use stark_engine_macros::stark_test;

use super::{
    super::super::tests::{finite_packed_codes, packed_f64_oracle, safe_iq_test_pipeline},
    *,
};

fn fixture(
    kind: u32,
    experts: usize,
    rows: usize,
    columns: usize,
) -> (MappedMatrix, Vec<u8>) {
    let codes = finite_packed_codes(kind, experts * rows, columns, 23);
    // Deliberately misalign the compressed byte view. The candidate only uses
    // byte loads; its device indices and F32 spans also have nonzero offsets.
    let mut guarded = vec![0xa5; 7];
    guarded.extend_from_slice(&codes);
    guarded.extend_from_slice(&[0xa5; 13]);
    let matrix = MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor().unwrap(), &guarded).unwrap(),
            offset: 7,
            length: codes.len(),
            registered_length: guarded.len(),
        },
        planes: None,
    };
    (matrix, codes)
}

fn ids(
    graph: &mut DecodeGraph,
    values: &[u32],
) -> DeviceVector {
    let mut guarded = vec![7.25];
    guarded.extend(values.iter().map(|&n| f32::from_bits(n)));
    guarded.push(7.25);
    graph.upload_into(&guarded).unwrap().slice(1, values.len()).unwrap()
}

#[expect(clippy::too_many_arguments, reason = "test-only explicit vector8 baseline avoids runtime schedule selectors")]
fn reference(
    graph: &mut DecodeGraph,
    matrix: &MappedMatrix,
    kind: u32,
    shape: [usize; 3],
    indices: &DeviceVector,
    selected: usize,
    input: &DeviceVector,
    tokens: usize,
    batched: bool,
) -> DeviceVector {
    let [experts, rows, columns] = shape;
    let output = graph.scratch(tokens * selected * rows).unwrap();
    let pipeline = safe_iq_test_pipeline(kind, "hybrid_graph_iq_indexed_vector8", 4);
    graph.check_finite(input).unwrap();
    for token in 0..tokens {
        let input = input
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
        let indices = indices.slice(token * selected, selected).unwrap();
        let row = output.slice(token * selected * rows, selected * rows).unwrap();
        graph.order_dispatch();
        graph.encoding.set_compute_pipeline_state(&pipeline);
        for (index, vector) in [&input, &row].into_iter().enumerate() {
            graph.encoding.set_buffer(Some(&vector.buffer), vector.offset, index + 1);
        }
        graph.encoding.set_buffer(Some(&matrix.codes.buffer), matrix.codes.offset, 0);
        let p = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
        graph.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        graph.encoding.set_buffer(Some(&indices.buffer), indices.offset, 6);
        graph.encoding.set_buffer(Some(&graph.failure.buffer), graph.failure.offset, 30);
        graph.encoding.dispatch_threadgroups(
            MTLSize {
                width: rows.div_ceil(8),
                height: selected,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        graph.resources.extend([input.buffer.clone(), indices.buffer.clone()]);
    }
    graph.resources.extend([matrix.codes.buffer.clone(), output.buffer.clone()]);
    graph.check_finite(&output).unwrap();
    output
}

#[stark_test]
fn iq_indexed_batch_preserves_vector8_bits_with_repeated_routes_and_tails() {
    let (experts, rows, selected) = (7usize, 13usize, 3usize);
    for kind in [17, 18] {
        for (tokens, columns) in [(1, 256), (5, 768), (8, 2048), (32, 256), (64, 256)] {
            let (matrix, codes) = fixture(kind, experts, rows, columns);
            let selections =
                (0..tokens * selected).map(|route| ((route * 3 + route / 5) % 5) as u32).collect::<Vec<_>>();
            for batched in [false, true] {
                let input_rows = if batched {
                    tokens * selected
                } else {
                    tokens
                };
                let values = (0..input_rows * columns).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>();
                let mut graph = DecodeGraph::new().unwrap();
                let input = graph.upload_into(&values).unwrap();
                let indices = ids(&mut graph, &selections);
                let expected = reference(
                    &mut graph,
                    &matrix,
                    kind,
                    [experts, rows, columns],
                    &indices,
                    selected,
                    &input,
                    tokens,
                    batched,
                );
                let actual = graph
                    .indexed_batch(
                        &matrix,
                        MatrixFormat::Packed(kind),
                        [experts, rows, columns],
                        &indices,
                        selected,
                        &input,
                        tokens,
                        batched,
                    )
                    .unwrap();
                let outputs = graph.finish(&[expected, actual]).unwrap();
                assert!(
                    outputs[0].iter().zip(&outputs[1]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "IQ batch changed vector8 bits kind={kind} tokens={tokens} batched={batched}"
                );
                if tokens <= 5 {
                    let expert_bytes = codes.len() / experts;
                    for (route, &expert) in selections.iter().enumerate() {
                        let expert = expert as usize;
                        let row = if batched {
                            route
                        } else {
                            route / selected
                        };
                        let oracle = packed_f64_oracle(
                            kind,
                            &codes[expert * expert_bytes..(expert + 1) * expert_bytes],
                            rows,
                            columns,
                            &values[row * columns..(row + 1) * columns],
                            0,
                        );
                        for (output_row, &(truth, absolute)) in oracle.iter().enumerate() {
                            assert!(
                                (f64::from(outputs[1][route * rows + output_row]) - truth).abs()
                                    <= 2e-6 * absolute.max(1.0),
                                "IQ batch F64 disagreement"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[stark_test]
fn iq_indexed_batch_preserves_maximum_routes_and_large_finite_inputs() {
    let (experts, rows, columns, tokens, selected) = (9, 5, 256, 64, 8);
    for kind in [17, 18] {
        let (matrix, _) = fixture(kind, experts, rows, columns);
        for batched in [false, true] {
            let selections = (0..tokens * selected).map(|route| (route % experts) as u32).collect::<Vec<_>>();
            let input_rows = if batched {
                tokens * selected
            } else {
                tokens
            };
            let values = (0..input_rows * columns)
                .map(|i| {
                    if i % 7 == 0 {
                        1e25
                    } else {
                        -1e24
                    }
                })
                .collect::<Vec<_>>();
            let mut graph = DecodeGraph::new().unwrap();
            let input = graph.upload_into(&values).unwrap();
            let indices = ids(&mut graph, &selections);
            let expected = reference(
                &mut graph,
                &matrix,
                kind,
                [experts, rows, columns],
                &indices,
                selected,
                &input,
                tokens,
                batched,
            );
            let actual = graph
                .indexed_batch(
                    &matrix,
                    MatrixFormat::Packed(kind),
                    [experts, rows, columns],
                    &indices,
                    selected,
                    &input,
                    tokens,
                    batched,
                )
                .unwrap();
            let outputs = graph.finish(&[expected, actual]).unwrap();
            assert!(outputs[0].iter().zip(&outputs[1]).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

#[stark_test]
fn iq_indexed_batch_matches_configured_projection_and_f64_at_model_widths() {
    let (experts, rows, tokens, selected) = (7usize, 13usize, 5usize, 3usize);
    for kind in [17, 18] {
        for columns in [2048usize, 4096] {
            let (_, codes) = fixture(kind, experts, rows, columns);
            let mut storage = vec![0xa5; 16];
            storage.extend_from_slice(&codes);
            let matrix = MappedMatrix {
                codes: MappedSlice {
                    buffer: buffer(executor().unwrap(), &storage).unwrap(),
                    offset: 16,
                    length: codes.len(),
                    registered_length: storage.len(),
                },
                planes: None,
            };
            let format = MatrixFormat::Packed(kind);
            // The usual model wrapper enables these selectors. Keep the test
            // valid under default CI flags while proving their actual admission
            // when the focused wrapper run requests them.
            for (name, admitted) in [
                ("STARK_GLM53_IQ_GPU_BOUND", iq_bound_schedule(format, columns)),
                ("STARK_GLM53_IQ_TYPED16", iq_typed16_schedule(format, columns, matrix.codes.offset)),
                ("STARK_GLM53_IQ_SIGNED_FACTOR", iq_signed_factor_schedule(format, columns, matrix.codes.offset)),
            ] {
                if std::env::var_os(name).is_some_and(|value| value == "1") {
                    assert!(admitted, "requested indexed schedule was not admitted: {name}");
                }
            }
            let selections = (0..tokens * selected).map(|i| ((i * 3 + i / 5) % 5) as u32).collect::<Vec<_>>();
            for batched in [false, true] {
                let input_rows = if batched {
                    tokens * selected
                } else {
                    tokens
                };
                let values = (0..columns * input_rows).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>();
                let mut graph = DecodeGraph::new().unwrap();
                let input = graph.upload_into(&values).unwrap();
                let indices = ids(&mut graph, &selections);
                let configured = graph.scratch(rows * tokens * selected).unwrap();
                for token in 0..tokens {
                    let source = graph
                        .indexed(
                            &matrix,
                            format,
                            [experts, rows, columns],
                            &indices.slice(token * selected, selected).unwrap(),
                            selected,
                            &input
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
                                .unwrap(),
                            batched,
                        )
                        .unwrap();
                    graph
                        .copy_into(&source, &configured.slice(token * selected * rows, selected * rows).unwrap())
                        .unwrap();
                }
                let candidate = graph
                    .indexed_batch(
                        &matrix,
                        format,
                        [experts, rows, columns],
                        &indices,
                        selected,
                        &input,
                        tokens,
                        batched,
                    )
                    .unwrap();
                let outputs = graph.finish(&[configured, candidate]).unwrap();
                let expert_bytes = codes.len() / experts;
                for (route, &expert) in selections.iter().enumerate() {
                    let expert = expert as usize;
                    let row = if batched {
                        route
                    } else {
                        route / selected
                    };
                    let oracle = packed_f64_oracle(
                        kind,
                        &codes[expert * expert_bytes..(expert + 1) * expert_bytes],
                        rows,
                        columns,
                        &values[row * columns..(row + 1) * columns],
                        0,
                    );
                    for (output_row, &(truth, absolute)) in oracle.iter().enumerate() {
                        let normalization = absolute.max(1.0);
                        let baseline = f64::from(outputs[0][route * rows + output_row]);
                        let candidate = f64::from(outputs[1][route * rows + output_row]);
                        assert!((baseline - truth).abs() / normalization <= 2e-6, "configured IQ F64 bound");
                        assert!((candidate - truth).abs() / normalization <= 2e-6, "batched IQ F64 bound");
                        assert!(
                            (candidate - baseline).abs() / normalization <= 2e-6,
                            "batched versus configured IQ bound"
                        );
                    }
                }
            }
        }
    }
}

#[stark_test]
fn iq_indexed_batch_integer_route_copy_preserves_metadata_before_projection() {
    let (experts, rows, columns, tokens, selected) = (289, 5, 256, 5, 3);
    let selections = [0u32, 1, 288, 257, 0, 1, 255, 256, 288, 1, 1, 1, 288, 0, 2];
    for kind in [17, 18] {
        let (matrix, _) = fixture(kind, experts, rows, columns);
        let mut graph = DecodeGraph::new().unwrap();
        let source = ids(&mut graph, &selections);
        let copied = graph.scratch(selections.len()).unwrap();
        graph
            .kernel("hybrid_graph_route_copy", &[&source, &copied], &[selections.len() as u32], selections.len(), 256)
            .unwrap();
        let input =
            graph.upload_into(&(0..columns * tokens).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>()).unwrap();
        let expected =
            reference(&mut graph, &matrix, kind, [experts, rows, columns], &source, selected, &input, tokens, false);
        let candidate = graph
            .indexed_batch(
                &matrix,
                MatrixFormat::Packed(kind),
                [experts, rows, columns],
                &copied,
                selected,
                &input,
                tokens,
                false,
            )
            .unwrap();
        let output = graph.finish(&[expected, candidate]).unwrap();
        assert!(output[0].iter().zip(&output[1]).all(|(a, b)| a.to_bits() == b.to_bits()));
        // The graph completion above fences the owned metadata writer. Read
        // integer bits directly; router IDs are not F32 values to validate.
        let copied_bits = unsafe {
            std::slice::from_raw_parts(
                copied.buffer.contents().as_ptr().cast::<u8>().add(copied.offset).cast::<u32>(),
                copied.len,
            )
        };
        assert_eq!(copied_bits, selections);
    }
}

#[stark_test]
fn iq_indexed_batch_rejects_invalid_routes_nonfinite_inputs_and_overflow() {
    let (experts, rows, columns, tokens, selected) = (7, 5, 256, 5, 3);
    for kind in [17, 18] {
        let (matrix, _) = fixture(kind, experts, rows, columns);
        for bad in [experts as u32, u32::MAX] {
            let mut graph = DecodeGraph::new().unwrap();
            let input = graph.upload_into(&vec![0.0; columns * tokens]).unwrap();
            let mut selections = vec![0; tokens * selected];
            selections[tokens * selected - 1] = bad;
            // u32::MAX has a NaN bit pattern: write device-owned metadata
            // directly rather than weaken the general F32 upload contract.
            let indices = DecodeGraph::zeros(selections.len()).unwrap();
            unsafe {
                indices
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u32>()
                    .copy_from_nonoverlapping(selections.as_ptr(), selections.len())
            };
            let output = graph
                .indexed_batch(
                    &matrix,
                    MatrixFormat::Packed(kind),
                    [experts, rows, columns],
                    &indices,
                    selected,
                    &input,
                    tokens,
                    false,
                )
                .unwrap();
            assert!(graph.finish(&[output]).is_err());
        }
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut graph = DecodeGraph::new().unwrap();
            let input = DecodeGraph::zeros(columns * tokens).unwrap();
            unsafe { input.buffer.contents().as_ptr().cast::<f32>().add(columns + 17).write(poison) };
            let indices = ids(&mut graph, &vec![0; tokens * selected]);
            let output = graph
                .indexed_batch(
                    &matrix,
                    MatrixFormat::Packed(kind),
                    [experts, rows, columns],
                    &indices,
                    selected,
                    &input,
                    tokens,
                    false,
                )
                .unwrap();
            // A later clamp must not mask a failed projection or input scan.
            let clipped = graph
                .swiglu(&output, &DecodeGraph::upload(&vec![1.0; rows * tokens * selected]).unwrap(), Some(1.0))
                .unwrap();
            assert!(graph.finish(&[clipped]).is_err());
        }
        let (_, mut huge_codes) = fixture(kind, experts, rows, columns);
        let (_, block_bytes) = weights::block_layout(kind).unwrap();
        for block in huge_codes.chunks_exact_mut(block_bytes) {
            block[..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        }
        let huge_matrix = MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), &huge_codes).unwrap(),
                offset: 0,
                length: huge_codes.len(),
                registered_length: huge_codes.len(),
            },
            planes: None,
        };
        let mut overflow = DecodeGraph::new().unwrap();
        let input = overflow.upload_into(&vec![f32::MAX; columns * tokens]).unwrap();
        let indices = ids(&mut overflow, &vec![0; tokens * selected]);
        let output = overflow
            .indexed_batch(
                &huge_matrix,
                MatrixFormat::Packed(kind),
                [experts, rows, columns],
                &indices,
                selected,
                &input,
                tokens,
                false,
            )
            .unwrap();
        let clipped = overflow
            .swiglu(&output, &DecodeGraph::upload(&vec![1.0; rows * tokens * selected]).unwrap(), Some(1.0))
            .unwrap();
        assert!(overflow.finish(&[clipped]).is_err());
        let mut graph = DecodeGraph::new().unwrap();
        let indices = ids(&mut graph, &[0]);
        let input = graph.upload_into(&vec![0.0; columns]).unwrap();
        for shape in [[0, rows, columns], [experts, usize::MAX, columns], [experts, rows, usize::MAX]] {
            assert!(
                graph.indexed_batch(&matrix, MatrixFormat::Packed(kind), shape, &indices, 1, &input, 1, false).is_err()
            );
        }
        assert!(
            graph
                .indexed_batch(
                    &matrix,
                    MatrixFormat::Packed(kind),
                    [experts, rows, columns],
                    &indices,
                    1,
                    &input,
                    65,
                    false
                )
                .is_err()
        );
    }
}

#[stark_test]
fn indexed_batch_fallback_preserves_existing_non_iq_projection() {
    let (experts, rows, columns, tokens, selected) = (3, 5, 32, 5, 2);
    let values = (0..experts * rows * columns).map(|i| ((i * 13 % 101) as f32 - 50.0) / 51.0).collect::<Vec<_>>();
    let matrix = MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor().unwrap(), bytemuck::cast_slice(&values)).unwrap(),
            offset: 0,
            length: values.len() * 4,
            registered_length: values.len() * 4,
        },
        planes: None,
    };
    for batched in [false, true] {
        let mut graph = DecodeGraph::new().unwrap();
        let input_rows = if batched {
            tokens * selected
        } else {
            tokens
        };
        let input = graph
            .upload_into(&(0..columns * input_rows).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>())
            .unwrap();
        let indices = ids(&mut graph, &(0..tokens * selected).map(|i| (i % experts) as u32).collect::<Vec<_>>());
        let mut expected = Vec::new();
        for token in 0..tokens {
            expected.push(
                graph
                    .indexed(
                        &matrix,
                        MatrixFormat::Packed(0),
                        [experts, rows, columns],
                        &indices.slice(token * selected, selected).unwrap(),
                        selected,
                        &input
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
                            .unwrap(),
                        batched,
                    )
                    .unwrap(),
            );
        }
        let actual = graph
            .indexed_batch(
                &matrix,
                MatrixFormat::Packed(0),
                [experts, rows, columns],
                &indices,
                selected,
                &input,
                tokens,
                batched,
            )
            .unwrap();
        expected.push(actual);
        let output = graph.finish(&expected).unwrap();
        assert!(output[..tokens].iter().flatten().zip(&output[tokens]).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

/// Component diagnostic only: synthetic route traffic, warm immutable bank,
/// complete projection including route packing and finite scans. It does not
/// qualify model throughput, host quietness or numerical equivalence by itself.
#[stark_test]
#[ignore]
fn iq_indexed_batch_gpu_component_timings() {
    let previous = profile::enabled();
    profile::enable(true);
    let mut results = Vec::new();
    for (kind, rows, columns) in [(17, 2048, 4096), (18, 2048, 4096), (17, 4096, 2048)] {
        let (experts, tokens, selected) = (288, 64, 8);
        let (matrix, _) = fixture(kind, experts, rows, columns);
        let selections = (0..tokens * selected)
            .map(|route| ((route * 13 + route / selected * 7) % experts) as u32)
            .collect::<Vec<_>>();
        for batched in [false, true] {
            let input_rows = if batched {
                tokens * selected
            } else {
                tokens
            };
            let input =
                DecodeGraph::upload(&(0..columns * input_rows).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>())
                    .unwrap();
            let indices =
                DecodeGraph::upload(&selections.iter().map(|&i| f32::from_bits(i)).collect::<Vec<_>>()).unwrap();
            let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            // Warm each arm once, then reverse the order in consecutive blocks.
            for iteration in 0..7 {
                let order = if iteration % 2 == 0 {
                    [0, 1, 2, 2, 1, 0]
                } else {
                    [2, 1, 0, 0, 1, 2]
                };
                for arm in order {
                    profile::reset();
                    let mut graph = DecodeGraph::new().unwrap();
                    if arm == 0 {
                        reference(
                            &mut graph,
                            &matrix,
                            kind,
                            [experts, rows, columns],
                            &indices,
                            selected,
                            &input,
                            tokens,
                            batched,
                        );
                    } else if arm == 1 {
                        for token in 0..tokens {
                            graph
                                .indexed(
                                    &matrix,
                                    MatrixFormat::Packed(kind),
                                    [experts, rows, columns],
                                    &indices.slice(token * selected, selected).unwrap(),
                                    selected,
                                    &input
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
                                        .unwrap(),
                                    batched,
                                )
                                .unwrap();
                        }
                    } else {
                        graph
                            .indexed_batch(
                                &matrix,
                                MatrixFormat::Packed(kind),
                                [experts, rows, columns],
                                &indices,
                                selected,
                                &input,
                                tokens,
                                batched,
                            )
                            .unwrap();
                    }
                    graph.finish(&[]).unwrap();
                    if iteration > 0 {
                        samples[arm].push(profile::snapshot()["native.metal.gpu"].seconds);
                    }
                }
            }
            for (arm, values) in samples.iter_mut().enumerate() {
                values.sort_by(f64::total_cmp);
                results.push(serde_json::json!({
                    "kind":kind, "rows":rows, "columns":columns, "experts":experts,
                    "tokens":tokens, "selected_per_token":selected, "input_batched":batched,
                    "arm":(["vector8_reference", "configured_indexed_reference", "grouped_four_route"][arm]),
                    "warm_samples":values.len(), "median_gpu_milliseconds":(values[5]+values[6])*500.0,
                    "minimum_gpu_milliseconds":values[0]*1000.0,
                    "maximum_gpu_milliseconds":values[values.len()-1]*1000.0,
                    "scope":"complete_projection_including_route_pack_and_finite_checks",
                    "model_throughput_qualified":false
                }));
            }
        }
    }
    profile::enable(previous);
    println!("{}", serde_json::json!({"iq_indexed_batch_component":results}));
}
