// SPDX-License-Identifier: Apache-2.0
//! Bitwise gates and a model-free timing probe for Edge0 token-batched rows.
//! Synthetic vectors stay in memory; the one-token kernels supply the expected
//! F32 bits. This is operator coverage only.

use stark_engine_macros::stark_test;

use super::*;

fn matrix(
    bytes: &[u8],
    offset: usize,
) -> MappedMatrix {
    let mut storage = vec![0x55; offset];
    storage.extend_from_slice(bytes);
    storage.extend_from_slice(&[0xaa; 7]);
    MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor().unwrap(), &storage).unwrap(),
            offset,
            length: bytes.len(),
            registered_length: storage.len(),
        },
        planes: None,
    }
}

fn packed(
    kind: u32,
    rows: usize,
    columns: usize,
    seed: usize,
) -> MappedMatrix {
    let (block, bytes) = weights::block_layout(kind).unwrap();
    let mut codes = (0..rows * (columns / block) * bytes).map(|i| (i * 37 + seed) as u8).collect::<Vec<_>>();
    for chunk in codes.chunks_exact_mut(bytes) {
        let scale = if kind == 11 {
            108
        } else {
            0
        };
        chunk[scale..scale + 2].copy_from_slice(&0x2800u16.to_le_bytes());
    }
    matrix(&codes, 7 + seed % 5)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn poisoned(
    len: usize,
    at: usize,
    poison: f32,
) -> DeviceVector {
    let input = DecodeGraph::zeros(len).unwrap();
    // SAFETY: fresh owned storage has no pending GPU users.
    unsafe {
        input.buffer.contents().as_ptr().cast::<f32>().add(at).write(poison);
    }
    input
}

#[stark_test]
fn edge0_token_rows_match_one_token_kernels_bitwise() {
    let rows = 13;
    let scale = DecodeGraph::upload(&(0..rows).map(|i| 0.5 + (i % 7) as f32 * 0.37).collect::<Vec<_>>()).unwrap();
    for columns in [512, 1792] {
        for count in [1usize, 2, 3, 5, 8, 9, 11] {
            // Token rows are scattered in a larger parent; outputs are permuted.
            let parents = count + 2;
            let input = (0..parents * columns).map(|i| ((i * 17 % 43) as f32 - 21.0) * 0.047).collect::<Vec<_>>();
            let source = DecodeGraph::upload(&input).unwrap();
            let input_rows = (0..count).map(|i| (i * 5 + 1) % parents).collect::<Vec<_>>();
            let output_rows = (0..count).map(|i| (i * 7 + 2) % count).collect::<Vec<_>>();
            let ordered = (0..count).collect::<Vec<_>>();
            for kind in [8, 11, 20] {
                let format = MatrixFormat::Packed(kind);
                let mapped = packed(kind, rows, columns, 19);
                let mut graph = DecodeGraph::new().unwrap();
                let mut pairs = Vec::new();
                for scaled in [false, true] {
                    let scale = scaled.then_some(&scale);
                    let expected = graph.scratch(rows * count).unwrap();
                    for (&from, &to) in input_rows.iter().zip(&output_rows) {
                        graph
                            .edge0_packed_single_into(
                                &mapped,
                                format,
                                rows,
                                columns,
                                &source.slice(from * columns, columns).unwrap(),
                                &expected.slice(to * rows, rows).unwrap(),
                                scale,
                                false,
                            )
                            .unwrap();
                    }
                    let actual = graph.scratch(rows * count).unwrap();
                    graph
                        .edge0_packed_tokens_into(
                            &mapped,
                            format,
                            rows,
                            columns,
                            &source,
                            &input_rows,
                            &actual,
                            &output_rows,
                            scale,
                        )
                        .unwrap();
                    pairs.push((expected, actual));
                }
                if count <= 8 {
                    // The original batched kernel on the leading rows.
                    let leading = source.slice(0, count * columns).unwrap();
                    let batched = graph
                        .edge0_packed_batch_selected(&mapped, format, rows, columns, &leading, count, false)
                        .unwrap();
                    let actual = graph.scratch(rows * count).unwrap();
                    graph
                        .edge0_packed_tokens_into(
                            &mapped, format, rows, columns, &source, &ordered, &actual, &ordered, None,
                        )
                        .unwrap();
                    pairs.push((batched, actual));
                }
                if kind == 11 {
                    let up = packed(kind, rows, columns, 73);
                    let fused =
                        graph.edge0_gate_up_swiglu_tokens(&mapped, &up, rows, columns, &source, &input_rows).unwrap();
                    for (index, &from) in input_rows.iter().enumerate() {
                        let single = graph
                            .edge0_gate_up_swiglu_single(
                                &mapped,
                                &up,
                                rows,
                                columns,
                                &source.slice(from * columns, columns).unwrap(),
                                false,
                            )
                            .unwrap();
                        pairs.push((single, fused.slice(index * rows, rows).unwrap()));
                    }
                    if count <= 8 {
                        let leading = source.slice(0, count * columns).unwrap();
                        let batched = graph
                            .edge0_gate_up_swiglu_batch_selected(&mapped, &up, rows, columns, &leading, count, false)
                            .unwrap();
                        let actual =
                            graph.edge0_gate_up_swiglu_tokens(&mapped, &up, rows, columns, &source, &ordered).unwrap();
                        pairs.push((batched, actual));
                    }
                }
                for (expected, actual) in &pairs {
                    graph.edge0_require_identical(expected, actual).unwrap();
                }
                let outputs = pairs.iter().flat_map(|(left, right)| [left.clone(), right.clone()]).collect::<Vec<_>>();
                let values = graph.finish(&outputs).unwrap();
                for pair in values.chunks_exact(2) {
                    assert_eq!(bits(&pair[0]), bits(&pair[1]), "Edge0 token rows changed an F32 result");
                    assert!(pair[0].iter().any(|value| *value != 0.0), "Edge0 token row fixture is degenerate");
                }
            }
        }
    }
}

#[stark_test]
fn edge0_token_rows_reject_poison_bad_scales_and_bad_shapes() {
    let scale = DecodeGraph::upload(&[1.5]).unwrap();
    let rows = [0usize, 1, 2];
    for kind in [8, 11, 20] {
        let format = MatrixFormat::Packed(kind);
        let mapped = packed(kind, 1, 256, 3);
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // Only the middle token row is poisoned, and nothing scans it first.
            for scaled in [false, true] {
                let input = poisoned(768, 256 + 255, poison);
                let mut graph = DecodeGraph::new().unwrap();
                let output = graph.scratch(3).unwrap();
                graph
                    .edge0_packed_tokens_into(
                        &mapped,
                        format,
                        1,
                        256,
                        &input,
                        &rows,
                        &output,
                        &rows,
                        scaled.then_some(&scale),
                    )
                    .unwrap();
                assert!(graph.finish(&[output]).is_err(), "Edge0 token rows concealed nonfinite input");
            }
            if kind == 11 {
                let input = poisoned(768, 256 + 255, poison);
                let mut graph = DecodeGraph::new().unwrap();
                let output = graph.edge0_gate_up_swiglu_tokens(&mapped, &mapped, 1, 256, &input, &rows).unwrap();
                assert!(graph.finish(&[output]).is_err(), "Edge0 fused token rows concealed nonfinite input");
            }
        }
        let input = DecodeGraph::upload(&[0.25; 768]).unwrap();
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let stored = poisoned(1, 0, bad);
            let mut graph = DecodeGraph::new().unwrap();
            let output = graph.scratch(3).unwrap();
            graph
                .edge0_packed_tokens_into(&mapped, format, 1, 256, &input, &rows, &output, &rows, Some(&stored))
                .unwrap();
            assert!(graph.finish(&[output]).is_err(), "Edge0 token rows accepted an invalid stored scale");
        }
        let mut graph = DecodeGraph::new().unwrap();
        let output = graph.scratch(3).unwrap();
        let wide_scale = DecodeGraph::upload(&[1.0, 1.0]).unwrap();
        let mut truncated = mapped.clone();
        truncated.codes.length -= 1;
        let mut attempt = |matrix: &MappedMatrix,
                           format: MatrixFormat,
                           from: &[usize],
                           to: &[usize],
                           scale: Option<&DeviceVector>| {
            graph.edge0_packed_tokens_into(matrix, format, 1, 256, &input, from, &output, to, scale).is_err()
        };
        assert!(attempt(&mapped, format, &[0, 3], &[0, 1], None), "input row beyond its parent");
        assert!(attempt(&mapped, format, &[0, 1], &[0, 3], None), "output row beyond its parent");
        assert!(attempt(&mapped, format, &[0, 1], &[1, 1], None), "two token rows share an output row");
        assert!(attempt(&mapped, format, &[0, 1], &[0], None), "row counts differ");
        assert!(attempt(&mapped, format, &[], &[], None), "empty token rows");
        assert!(attempt(&truncated, format, &[0, 1], &[0, 1], None), "truncated weight span");
        assert!(attempt(&mapped, MatrixFormat::Packed(30), &[0, 1], &[0, 1], None), "unsupported format");
        assert!(attempt(&mapped, format, &[0, 1], &[0, 1], Some(&wide_scale)), "stored scale shape");
        let shared = graph.scratch(768).unwrap();
        assert!(
            graph
                .edge0_packed_tokens_into(
                    &mapped,
                    format,
                    1,
                    256,
                    &shared.slice(0, 512).unwrap(),
                    &[0, 1],
                    &shared.slice(511, 2).unwrap(),
                    &[0, 1],
                    None,
                )
                .is_err(),
            "Edge0 token rows accepted an aliased output"
        );
        assert!(
            graph
                .edge0_packed_tokens_into(
                    &mapped,
                    format,
                    1,
                    256,
                    &input.slice(0, 700).unwrap(),
                    &[0, 1],
                    &output,
                    &[0, 1],
                    None,
                )
                .is_err(),
            "Edge0 token rows accepted a ragged parent"
        );
        if kind == 11 {
            assert!(graph.edge0_gate_up_swiglu_tokens(&mapped, &truncated, 1, 256, &input, &[0, 1]).is_err());
            assert!(graph.edge0_gate_up_swiglu_tokens(&mapped, &mapped, 1, 256, &input, &[0, 3]).is_err());
            assert!(graph.edge0_gate_up_swiglu_tokens(&mapped, &mapped, 1, 256, &input, &[]).is_err());
        }
        graph.finish(&[]).unwrap();
    }
}

/// A float matrix of small dyadic values, exact in F32, F16 and BF16.
fn floats(
    kind: u32,
    rows: usize,
    columns: usize,
    seed: usize,
) -> MappedMatrix {
    let mut bytes = Vec::new();
    for index in 0..rows * columns {
        let value = ((index * 37 + seed) % 61) as f32 / 64.0 - 0.46875;
        let bits = value.to_bits();
        match kind {
            0 => bytes.extend_from_slice(&bits.to_le_bytes()),
            30 => bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes()),
            _ => {
                // IEEE half of a normal value with at most ten mantissa bits.
                let exponent = ((bits >> 23) & 0xff) as i32 - 112;
                assert!(value == 0.0 || ((1..31).contains(&exponent) && bits & 0x1fff == 0));
                let half = if value == 0.0 {
                    0
                } else {
                    ((bits >> 16) & 0x8000) as u16 | (exponent as u16) << 10 | ((bits >> 13) & 0x3ff) as u16
                };
                bytes.extend_from_slice(&half.to_le_bytes());
            },
        }
    }
    matrix(&bytes, 8 + 4 * (seed % 3))
}

/// The hyper-connection and mixer output projections replace the generic
/// one-token projection, so that projection supplies the expected bits.
#[stark_test]
fn edge0_token_rows_match_generic_one_token_projection_bitwise() {
    let shapes =
        [(0, 9, 320), (1, 9, 320), (30, 9, 320), (0, 5, 1024), (1, 7, 2560), (1, 320, 512), (8, 9, 320), (8, 5, 6144)];
    for (kind, rows, columns) in shapes {
        let format = MatrixFormat::Packed(kind);
        let mapped = if kind == 8 {
            packed(kind, rows, columns, 23)
        } else {
            floats(kind, rows, columns, 23)
        };
        for count in [1usize, 2, 3, 8] {
            let input = (0..count * columns).map(|i| ((i * 13 % 47) as f32 - 23.0) * 0.031).collect::<Vec<_>>();
            let source = DecodeGraph::upload(&input).unwrap();
            let mut graph = DecodeGraph::new().unwrap();
            let batched = graph.edge0_matvec_tokens(&mapped, format, rows, columns, &source, count).unwrap();
            let mut pairs = Vec::new();
            for token in 0..count {
                let reference = graph
                    .mapped(
                        &MappedOperation {
                            matrices: vec![mapped.clone()],
                            inputs: &[],
                            rows,
                            columns,
                            rows_per_batch: 0,
                            format,
                        },
                        &source.slice(token * columns, columns).unwrap(),
                    )
                    .unwrap();
                pairs.push((reference, batched.slice(token * rows, rows).unwrap()));
            }
            // The original batched kernel accepts the same formats.
            let original =
                graph.edge0_packed_batch_selected(&mapped, format, rows, columns, &source, count, false).unwrap();
            pairs.push((original, batched.clone()));
            for (expected, actual) in &pairs {
                graph.edge0_require_identical(expected, actual).unwrap();
            }
            let outputs = pairs.iter().flat_map(|(left, right)| [left.clone(), right.clone()]).collect::<Vec<_>>();
            let values = graph.finish(&outputs).unwrap();
            for pair in values.chunks_exact(2) {
                assert_eq!(bits(&pair[0]), bits(&pair[1]), "Edge0 token rows changed a generic projection");
                assert!(pair[0].iter().any(|value| *value != 0.0), "Edge0 generic projection fixture is degenerate");
            }
        }
        // A poisoned token row is rejected without any separate scan.
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let input = poisoned(3 * columns, columns + columns - 1, poison);
            let mut graph = DecodeGraph::new().unwrap();
            let output = graph.edge0_matvec_tokens(&mapped, format, rows, columns, &input, 3).unwrap();
            assert!(graph.finish(&[output]).is_err(), "Edge0 generic token rows concealed nonfinite input");
        }
        let input = DecodeGraph::upload(&vec![0.25; 2 * columns]).unwrap();
        let scale = DecodeGraph::upload(&vec![1.0; rows]).unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        let output = graph.scratch(2 * rows).unwrap();
        if kind != 8 {
            assert!(
                graph
                    .edge0_packed_tokens_into(
                        &mapped,
                        format,
                        rows,
                        columns,
                        &input,
                        &[0, 1],
                        &output,
                        &[0, 1],
                        Some(&scale),
                    )
                    .is_err(),
                "Edge0 float token rows accepted a stored scale"
            );
        }
        assert!(graph.edge0_matvec_tokens(&mapped, format, rows, columns, &input, 3).is_err());
        graph.finish(&[]).unwrap();
    }
}

/// A token batch above eight takes several dispatches. The original batched
/// kernel, eight tokens at a time, supplies the expected bits.
#[stark_test]
fn edge0_token_batches_above_eight_match_eight_token_batches_bitwise() {
    let (rows, columns) = (11, 512);
    for kind in [0, 1, 8, 11, 20, 30] {
        let format = MatrixFormat::Packed(kind);
        let mapped = if matches!(kind, 0 | 1 | 30) {
            floats(kind, rows, columns, 29)
        } else {
            packed(kind, rows, columns, 29)
        };
        for count in [9usize, 16, 17, 32] {
            let input = (0..count * columns).map(|i| ((i * 19 % 53) as f32 - 26.0) * 0.023).collect::<Vec<_>>();
            let source = DecodeGraph::upload(&input).unwrap();
            let mut graph = DecodeGraph::new().unwrap();
            let batched = graph.edge0_packed_tokens(&mapped, format, rows, columns, &source, count).unwrap();
            let unscanned = graph.edge0_matvec_tokens(&mapped, format, rows, columns, &source, count).unwrap();
            let mut pairs = vec![(batched.clone(), unscanned)];
            for first in (0..count).step_by(8) {
                let tokens = 8.min(count - first);
                let reference = graph
                    .edge0_packed_batch_selected(
                        &mapped,
                        format,
                        rows,
                        columns,
                        &source.slice(first * columns, tokens * columns).unwrap(),
                        tokens,
                        false,
                    )
                    .unwrap();
                pairs.push((reference, batched.slice(first * rows, tokens * rows).unwrap()));
            }
            for (expected, actual) in &pairs {
                graph.edge0_require_identical(expected, actual).unwrap();
            }
            let outputs = pairs.iter().flat_map(|(left, right)| [left.clone(), right.clone()]).collect::<Vec<_>>();
            let values = graph.finish(&outputs).unwrap();
            for pair in values.chunks_exact(2) {
                assert_eq!(bits(&pair[0]), bits(&pair[1]), "Edge0 larger token batch changed an F32 result");
                assert!(pair[0].iter().any(|value| *value != 0.0), "Edge0 larger token batch fixture is degenerate");
            }
        }
        let beyond = DecodeGraph::upload(&vec![0.25; (EDGE0_TOKEN_BATCH_LIMIT + 1) * columns]).unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        assert!(
            graph.edge0_packed_tokens(&mapped, format, rows, columns, &beyond, EDGE0_TOKEN_BATCH_LIMIT + 1).is_err()
        );
        assert!(
            graph.edge0_matvec_tokens(&mapped, format, rows, columns, &beyond, EDGE0_TOKEN_BATCH_LIMIT + 1).is_err()
        );
        graph.finish(&[]).unwrap();
    }

    // The ordered route reduction is per token; sixteen tokens must equal two
    // batches of eight.
    let (width, selected, tokens) = (64usize, 3usize, 16usize);
    let upload = |len: usize, step: usize| {
        DecodeGraph::upload(&(0..len).map(|i| ((i * step % 37) as f32 - 18.0) * 0.019).collect::<Vec<_>>()).unwrap()
    };
    let experts = upload(tokens * selected * width, 7);
    let probabilities = upload(tokens * selected, 11);
    let shared = upload(tokens * width, 13);
    let gate = upload(tokens, 17);
    let mut graph = DecodeGraph::new().unwrap();
    let whole = graph.edge0_weighted_sum(&experts, &probabilities, &shared, &gate, width, selected, tokens).unwrap();
    let mut outputs = vec![whole.clone()];
    for first in [0usize, 8] {
        let reference = graph
            .edge0_weighted_sum(
                &experts.slice(first * selected * width, 8 * selected * width).unwrap(),
                &probabilities.slice(first * selected, 8 * selected).unwrap(),
                &shared.slice(first * width, 8 * width).unwrap(),
                &gate.slice(first, 8).unwrap(),
                width,
                selected,
                8,
            )
            .unwrap();
        graph.edge0_require_identical(&whole.slice(first * width, 8 * width).unwrap(), &reference).unwrap();
        outputs.push(reference);
    }
    let values = graph.finish(&outputs).unwrap();
    assert_eq!(bits(&values[0][..8 * width]), bits(&values[1]));
    assert_eq!(bits(&values[0][8 * width..]), bits(&values[2]));
}

/// GPU seconds of one command holding `encode`'s dispatches.
fn gpu_seconds(encode: impl FnOnce(&mut DecodeGraph)) -> f64 {
    let mut graph = DecodeGraph::new().unwrap();
    graph.set_scratch_limit(256 * 1024 * 1024).unwrap();
    encode(&mut graph);
    let mut pending = graph.submit(&[]).unwrap();
    pending.complete().unwrap();
    pending.graph.command.gpu_end_time() - pending.graph.command.gpu_start_time()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Model-free timing probe, not a qualification. It prints microseconds of
/// command GPU time per token row for one-token dispatches, the original
/// batched kernel and the token-batched kernel on Edge0's matrix shapes.
#[stark_test]
#[ignore]
fn edge0_token_rows_timing_probe() {
    const TOKEN_ROWS: usize = 512;
    const REPEATS: usize = 7;
    // name, kind, rows, columns, distinct matrices, fused gate/up, stored scale
    let shapes = [
        ("u_q8_1792_to_2560", 8, 2560, 1792, 8, false, false),
        ("d_q8_2560_to_1792_scaled", 8, 1792, 2560, 8, false, true),
        ("qkv_q8_2560_to_10240", 8, 10240, 2560, 2, false, false),
        ("shared_q8_2560_to_640", 8, 640, 2560, 4, false, false),
        ("expert_gate_up_q3_1792_to_640", 11, 640, 1792, 16, true, false),
        ("expert_down_iq4_640_to_1792", 20, 1792, 640, 16, false, false),
    ];
    for (name, kind, rows, columns, count, fused, scaled) in shapes {
        let format = MatrixFormat::Packed(kind);
        let matrices = (0..count).map(|seed| packed(kind, rows, columns, 11 + seed)).collect::<Vec<_>>();
        let partners = (0..if fused {
            count
        } else {
            0
        })
            .map(|seed| packed(kind, rows, columns, 101 + seed))
            .collect::<Vec<_>>();
        let scale = DecodeGraph::upload(&(0..rows).map(|i| 0.75 + (i % 11) as f32 * 0.21).collect::<Vec<_>>()).unwrap();
        let scale = scaled.then_some(&scale);
        let input = (0..8 * columns).map(|i| ((i * 29 % 61) as f32 - 30.0) * 0.011).collect::<Vec<_>>();
        let source = DecodeGraph::upload(&input).unwrap();
        for tokens in [2usize, 4, 8] {
            let ordered = (0..tokens).collect::<Vec<_>>();
            let passes = TOKEN_ROWS / (count * tokens);
            let singles = |graph: &mut DecodeGraph| {
                let output = graph.scratch(rows * tokens).unwrap();
                for _ in 0..passes {
                    for (index, mapped) in matrices.iter().enumerate() {
                        for token in 0..tokens {
                            let row = source.slice(token * columns, columns).unwrap();
                            if fused {
                                graph
                                    .edge0_gate_up_swiglu_single(mapped, &partners[index], rows, columns, &row, false)
                                    .unwrap();
                            } else {
                                graph
                                    .edge0_packed_single_into(
                                        mapped,
                                        format,
                                        rows,
                                        columns,
                                        &row,
                                        &output.slice(token * rows, rows).unwrap(),
                                        scale,
                                        false,
                                    )
                                    .unwrap();
                            }
                        }
                    }
                }
            };
            let original = |graph: &mut DecodeGraph| {
                let leading = source.slice(0, tokens * columns).unwrap();
                let output = graph.scratch(rows * tokens).unwrap();
                for _ in 0..passes {
                    for (index, mapped) in matrices.iter().enumerate() {
                        if fused {
                            graph
                                .edge0_dispatch(
                                    "edge0_graph_gate_up_swiglu_batch",
                                    &[mapped, &partners[index]],
                                    &[&leading, &output],
                                    &[rows as u32, columns as u32, tokens as u32],
                                    rows,
                                    false,
                                )
                                .unwrap();
                        } else {
                            let (block, bytes) = weights::block_layout(kind).unwrap();
                            graph
                                .edge0_dispatch(
                                    "edge0_graph_packed_batch",
                                    &[mapped],
                                    &[&leading, &output],
                                    &[rows as u32, columns as u32, kind, block as u32, bytes as u32, tokens as u32],
                                    rows,
                                    false,
                                )
                                .unwrap();
                            if let Some(scale) = scale {
                                graph.edge0_column_scale(&output, scale, rows, tokens).unwrap();
                            }
                        }
                    }
                }
            };
            let batched = |graph: &mut DecodeGraph| {
                let output = graph.scratch(rows * tokens).unwrap();
                for _ in 0..passes {
                    for (index, mapped) in matrices.iter().enumerate() {
                        if fused {
                            graph
                                .edge0_gate_up_swiglu_tokens(mapped, &partners[index], rows, columns, &source, &ordered)
                                .unwrap();
                        } else {
                            graph
                                .edge0_packed_tokens_into(
                                    mapped, format, rows, columns, &source, &ordered, &output, &ordered, scale,
                                )
                                .unwrap();
                        }
                    }
                }
            };
            // One untimed pass builds pipelines and touches every page.
            gpu_seconds(singles);
            gpu_seconds(original);
            gpu_seconds(batched);
            let mut seconds = [Vec::new(), Vec::new(), Vec::new()];
            for _ in 0..REPEATS {
                seconds[0].push(gpu_seconds(singles));
                seconds[1].push(gpu_seconds(original));
                seconds[2].push(gpu_seconds(batched));
            }
            let per_row = |values: &Vec<f64>| median(values.clone()) * 1e6 / (passes * count * tokens) as f64;
            let (single, old, new) = (per_row(&seconds[0]), per_row(&seconds[1]), per_row(&seconds[2]));
            println!(
                "edge0_token_probe shape={name} tokens={tokens} rows_timed={} one_token_us={single:.2} \
                 original_batch_us={old:.2} token_batch_us={new:.2} speedup_vs_one_token={:.2}",
                passes * count * tokens,
                single / new,
            );
        }
    }
}
