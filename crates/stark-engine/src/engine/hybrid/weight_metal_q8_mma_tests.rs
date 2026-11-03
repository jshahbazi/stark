// SPDX-License-Identifier: Apache-2.0
//! Synthetic, model-free gates for the opt-in GLM Q8_0/T8 F32 MMA candidate.
//! The shared MMA schedule retains its existing MIT llama.cpp revision and
//! attribution in packed_kquant_mma_matvec.metal; this adds no upstream code.
//! No weights, activations or numerical output vectors are printed or saved.

use stark_engine_macros::stark_test;

use super::{
    tests::{finite_packed_codes, packed_f64_oracle},
    *,
};

const ACTUAL_SHAPES: [(usize, usize); 4] = [(8192, 4096), (16384, 1536), (4096, 1536), (512, 4096)];
const COUNTER: &str = "native.glm.prefill.q8_mma";

fn qualified() -> bool {
    std::env::var_os("STARK_GLM53_PREFILL_Q8_MMA").is_some_and(|value| value == "1")
        && executor().unwrap().device.name() == "Apple M5 Max"
        && executor().unwrap().packed_fast_library.is_none()
}

fn make_operation(
    codes: &[u8],
    rows: usize,
    columns: usize,
) -> MappedOperation<'static> {
    // Odd byte offset deliberately excludes aligned half-pointer shortcuts.
    let mut padded = vec![0xa7u8; 19];
    padded.extend_from_slice(codes);
    padded.extend_from_slice(&[0x4d; 19]);
    MappedOperation {
        matrices: vec![MappedMatrix {
            codes: MappedSlice {
                buffer: buffer(executor().unwrap(), &padded).unwrap(),
                offset: 19,
                length: codes.len(),
                registered_length: padded.len(),
            },
            planes: None,
        }],
        inputs: &[],
        rows,
        columns,
        rows_per_batch: 0,
        format: MatrixFormat::Packed(8),
    }
}

fn input_view(values: &[f32]) -> DeviceVector {
    // NaN guards prove that finite scans honor the actual input view.
    let mut padded = vec![f32::NAN; 3];
    padded.extend_from_slice(values);
    padded.extend_from_slice(&[f32::NAN; 3]);
    DeviceVector {
        buffer: buffer(executor().unwrap(), bytemuck::cast_slice(&padded)).unwrap(),
        offset: 12,
        len: values.len(),
    }
}

fn run(
    operation: &MappedOperation<'_>,
    values: &[f32],
    tokens: usize,
    mma: bool,
) -> Result<(Vec<f32>, u64)> {
    let previous = profile::enabled();
    profile::enable(true);
    let before = profile::snapshot().get(COUNTER).map_or(0, |value| value.calls);
    let result = (|| {
        let mut graph = DecodeGraph::new()?;
        let before_guard = graph.upload_into(&[7.25; 37])?;
        let output = graph.mapped_same_weight(operation, &input_view(values), tokens, mma)?;
        assert!(output.offset > 0, "Q8 MMA output-offset probe missed its scratch view");
        let after_guard = graph.upload_into(&[7.25; 37])?;
        let mut outputs = graph.finish(&[before_guard, output, after_guard])?;
        assert!(outputs[0].iter().chain(&outputs[2]).all(|&value| value == 7.25), "Q8 MMA output guard changed");
        Ok(outputs.swap_remove(1))
    })();
    let after = profile::snapshot().get(COUNTER).map_or(0, |value| value.calls);
    profile::enable(previous);
    result.map(|values| (values, after - before))
}

fn configured_source(
    codes: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(input.len() / columns * rows);
    for token in input.chunks_exact(columns) {
        output.extend(matvec(codes, 8, rows, columns, token)?);
    }
    Ok(output)
}

fn same_bits(
    actual: &[f32],
    source: &[f32],
) -> bool {
    actual.len() == source.len() && actual.iter().zip(source).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[expect(clippy::too_many_arguments, reason = "explicit dimensions and independent source/F64 numerical gates")]
fn check_numerics(
    codes: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    actual: &[f32],
    source: &[f32],
    sample_rows: usize,
) {
    assert!(actual.len() == source.len() && actual.len() == rows * input.len() / columns, "Q8 MMA result shape");
    let squared_reference = source.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>();
    let squared_error = actual.iter().zip(source).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum::<f64>();
    assert!((squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "Q8 MMA configured source RMS");
    let row_bytes = columns / 32 * 34;
    let samples = rows.min(sample_rows).max(1);
    for (token, values) in input.chunks_exact(columns).enumerate() {
        for sample in 0..samples {
            let row = if samples == 1 {
                0
            } else {
                sample * (rows - 1) / (samples - 1)
            };
            let (expected, absolute) =
                packed_f64_oracle(8, &codes[row * row_bytes..(row + 1) * row_bytes], 1, columns, values, 0)[0];
            // Q8_0's half scale and signed-byte product are exactly F32
            // representable, so this is also the mathematical GGUF oracle.
            for value in [actual[token * rows + row], source[token * rows + row]] {
                assert!(
                    (f64::from(value) - expected).abs() <= 2e-6 * absolute.max(1.0),
                    "Q8 MMA/source F64 normalized error"
                );
            }
        }
    }
}

#[stark_test]
fn q8_mma_actual_shapes_configured_source_f64_and_offsets() {
    for (rows, columns) in ACTUAL_SHAPES {
        let codes = finite_packed_codes(8, rows, columns, rows + columns + 41);
        let input = (0..8 * columns)
            .map(|index| (((index * 31 + index / columns * 17) % 149) as f32 - 74.0) * 0.0078125)
            .collect::<Vec<_>>();
        let source = configured_source(&codes, rows, columns, &input).unwrap();
        let operation = make_operation(&codes, rows, columns);
        let (actual, admitted) = run(&operation, &input, 8, true).unwrap();
        assert_eq!(admitted, u64::from(qualified()), "Q8 MMA actual-shape admission counter");
        check_numerics(&codes, rows, columns, &input, &actual, &source, 17);
        let (serial, admitted) = run(&operation, &input, 8, false).unwrap();
        assert!(admitted == 0 && same_bits(&serial, &source), "Q8 MMA disabled wrapper changed configured source bits");
        if !qualified() {
            assert!(same_bits(&actual, &source), "Q8 MMA unqualified route changed source bits");
        }
    }
}

#[stark_test]
fn q8_mma_independent_token_basis_crosses_16_and_32_value_boundaries() {
    let (rows, columns) = (512, 4096);
    let codes = finite_packed_codes(8, rows, columns, 59);
    let mut input = vec![0.0f32; 8 * columns];
    let basis = [0, 15, 16, 31, 32, 63, 64, columns - 1];
    for (token, &column) in basis.iter().enumerate() {
        input[token * columns + column] = if token % 2 == 0 {
            0.5
        } else {
            -2.0
        };
    }
    let (actual, admitted) = run(&make_operation(&codes, rows, columns), &input, 8, true).unwrap();
    assert_eq!(admitted, u64::from(qualified()), "Q8 basis route did not match admission");
    let row_bytes = columns / 32 * 34;
    for (token, &column) in basis.iter().enumerate() {
        for row in 0..rows {
            let at = row * row_bytes + column / 32 * 34;
            let mut decoded = [0.0f32; 32];
            weights::decode_block(8, &codes[at..at + 34], &mut decoded).unwrap();
            let expected = decoded[column % 32] * input[token * columns + column];
            assert!(actual[token * rows + row] == expected, "Q8 independent token/row basis mapping changed");
        }
    }
}

#[stark_test]
fn q8_mma_tail_unsupported_shape_and_invalid_view_admission() {
    for (rows, columns, token_counts) in
        [(512, 4096, vec![1, 2, 3, 4, 5, 6, 7]), (513, 4096, vec![8]), (37, 96, vec![8])]
    {
        let codes = finite_packed_codes(8, rows, columns, 71);
        let operation = make_operation(&codes, rows, columns);
        for tokens in token_counts {
            let input = (0..tokens * columns).map(|index| (index as f32 * 0.017).sin()).collect::<Vec<_>>();
            let source = configured_source(&codes, rows, columns, &input).unwrap();
            let (actual, admitted) = run(&operation, &input, tokens, true).unwrap();
            assert!(
                admitted == 0 && same_bits(&actual, &source),
                "Q8 MMA tail/unsupported shape changed configured source bits"
            );
        }
    }
    let (rows, columns) = (512, 4096);
    let codes = finite_packed_codes(8, rows, columns, 79);
    let input = vec![0.25; 8 * columns];
    for tokens in [0, 9] {
        let mut graph = DecodeGraph::new().unwrap();
        assert!(
            graph
                .mapped_same_weight(&make_operation(&codes, rows, columns), &input_view(&input), tokens, true)
                .is_err(),
            "Q8 MMA invalid token count admitted"
        );
    }
    let mut short = make_operation(&codes, rows, columns);
    short.matrices[0].codes.length -= 1;
    assert!(run(&short, &input, 8, true).is_err(), "Q8 MMA truncated weight view admitted");
    let mut out_of_bounds = make_operation(&codes, rows, columns);
    out_of_bounds.matrices[0].codes.offset = out_of_bounds.matrices[0].codes.buffer.length() - 1;
    assert!(run(&out_of_bounds, &input, 8, true).is_err(), "Q8 MMA weight view outside buffer admitted");
    let mut graph = DecodeGraph::new().unwrap();
    let mut unaligned = input_view(&input);
    unaligned.offset += 2;
    assert!(
        graph.mapped_same_weight(&make_operation(&codes, rows, columns), &unaligned, 8, true).is_err(),
        "Q8 MMA unaligned F32 view admitted"
    );
}

#[stark_test]
fn q8_mma_extremes_preserve_configured_source_bits_on_every_actual_shape() {
    for (rows, columns) in ACTUAL_SHAPES {
        let codes = finite_packed_codes(8, rows, columns, 83);
        let operation = make_operation(&codes, rows, columns);
        let input = (0..8 * columns)
            .map(|index| {
                if (index * 17 + index / columns) % 5 < 2 {
                    1.0e21
                } else {
                    -1.0e21
                }
            })
            .collect::<Vec<_>>();
        let source = configured_source(&codes, rows, columns, &input).unwrap();
        let (actual, admitted) = run(&operation, &input, 8, true).unwrap();
        assert_eq!(admitted, u64::from(qualified()), "Q8 extreme route did not match admission");
        assert!(same_bits(&actual, &source), "Q8 MMA extreme fallback changed configured source chronology");
    }
}

#[stark_test]
fn q8_mma_nonfinite_sticky_status_and_zero_times_max_contract() {
    let (rows, columns) = (512, 4096);
    let codes = finite_packed_codes(8, rows, columns, 89);
    let operation = make_operation(&codes, rows, columns);
    let finite = vec![0.25; 8 * columns];
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = finite.clone();
        input[7 * columns + 31] = poison;
        let mut graph = DecodeGraph::new().unwrap();
        let _bad = graph.mapped_same_weight(&operation, &input_view(&input), 8, true).unwrap();
        let good = graph.mapped_same_weight(&operation, &input_view(&finite), 8, true).unwrap();
        assert!(graph.finish(&[good]).is_err(), "Q8 MMA subsequent finite output cleared sticky input failure");
    }
    for poison in [half::f16::NAN, half::f16::INFINITY, half::f16::NEG_INFINITY] {
        let mut bad_codes = codes.clone();
        bad_codes[..2].copy_from_slice(&poison.to_bits().to_le_bytes());
        let mut graph = DecodeGraph::new().unwrap();
        let _bad = graph
            .mapped_same_weight(&make_operation(&bad_codes, rows, columns), &input_view(&finite), 8, true)
            .unwrap();
        let good = graph.mapped_same_weight(&operation, &input_view(&finite), 8, true).unwrap();
        assert!(graph.finish(&[good]).is_err(), "Q8 MMA subsequent finite output cleared sticky weight failure");
    }
    let mut zero_block = vec![0u8; 34];
    zero_block[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
    let zero_codes = zero_block.repeat(rows * columns / 32);
    let (actual, admitted) =
        run(&make_operation(&zero_codes, rows, columns), &vec![f32::MAX; 8 * columns], 8, true).unwrap();
    assert_eq!(admitted, u64::from(qualified()), "Q8 zero-times-MAX route did not match admission");
    assert!(actual.iter().all(|&value| value == 0.0), "Q8 MMA zero times MAX contract changed");
    let mut overflow_block = vec![1u8; 34];
    overflow_block[..2].copy_from_slice(&half::f16::MAX.to_bits().to_le_bytes());
    let overflow = overflow_block.repeat(rows * columns / 32);
    assert!(
        run(&make_operation(&overflow, rows, columns), &vec![f32::MAX; 8 * columns], 8, true).is_err(),
        "Q8 MMA concealed finite-input overflow"
    );
}

/// Complete synthetic T8 projection commands with finite scans. The only
/// printed values are aggregate GPU command durations and matrix dimensions.
/// These component timings do not qualify full-model prefill throughput.
#[stark_test]
#[ignore]
fn q8_mma_actual_shape_gpu_component_timings() {
    assert!(qualified(), "Q8 timing requires Safe source, Apple M5 Max and STARK_GLM53_PREFILL_Q8_MMA=1");
    let previous = profile::enabled();
    profile::enable(true);
    let mut results = Vec::new();
    for (rows, columns) in ACTUAL_SHAPES {
        let codes = finite_packed_codes(8, rows, columns, rows + columns + 97);
        let operation = make_operation(&codes, rows, columns);
        let values = (0..8 * columns).map(|index| ((index * 31 % 149) as f32 - 74.0) * 0.0078125).collect::<Vec<_>>();
        let input = DecodeGraph::upload(&values).unwrap();
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        // Two warmup pairs, then twelve pairs with reversed arm order.
        for iteration in 0..14 {
            let order = if iteration % 2 == 0 {
                [0, 1]
            } else {
                [1, 0]
            };
            for arm in order {
                profile::reset();
                let mut graph = DecodeGraph::new().unwrap();
                let output = graph.mapped_same_weight(&operation, &input, 8, arm == 1).unwrap();
                graph.finish(&[output]).unwrap();
                let snapshot = profile::snapshot();
                let admitted = snapshot.get(COUNTER).map_or(0, |value| value.calls);
                assert_eq!(admitted, u64::from(arm == 1), "Q8 timing arm admission mismatch");
                let seconds = snapshot.get("native.metal.gpu").map_or(0.0, |value| value.seconds);
                assert!(seconds.is_finite() && seconds > 0.0, "Q8 timing missing completed GPU command duration");
                if iteration >= 2 {
                    samples[arm].push(seconds);
                }
            }
        }
        for (arm, values) in samples.iter_mut().enumerate() {
            assert_eq!(values.len(), 12);
            values.sort_by(f64::total_cmp);
            results.push(serde_json::json!({
                "kind": 8, "rows": rows, "columns": columns, "tokens": 8,
                "arm": if arm == 0 { "configured_serial_source" } else { "q8_f32_mma" },
                "warm_samples": values.len(),
                "median_gpu_milliseconds": (values[5] + values[6]) * 500.0,
                "minimum_gpu_milliseconds": values[0] * 1000.0,
                "maximum_gpu_milliseconds": values[11] * 1000.0,
                "timer_scope": "complete_projection_including_finite_checks",
                "model_throughput_qualified": false
            }));
        }
    }
    profile::enable(previous);
    println!("{}", serde_json::json!({"q8_mma_component": results}));
}
