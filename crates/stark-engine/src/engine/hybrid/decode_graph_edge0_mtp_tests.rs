// SPDX-License-Identifier: Apache-2.0
//! Synthetic original-plane tests. F64 references are computed from dense
//! logical weights generated before packing, without decoding packed bytes.
//! These tests qualify staged operators only, never a real MTP artifact.

use stark_engine_macros::stark_test;

use super::*;

fn mapped(
    bytes: &[u8],
    offset: usize,
) -> MappedMatrix {
    let mut storage = vec![0x97; offset];
    storage.extend_from_slice(bytes);
    storage.extend_from_slice(&[0xa5; 13]);
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

struct Fixture {
    bytes: Vec<u8>,
    gate: Vec<f64>,
    up: Vec<f64>,
    down: Vec<f64>,
}

impl Fixture {
    fn new(
        hidden: usize,
        intermediate: usize,
    ) -> Self {
        // Construct each logical row first; collect all code rows separately
        // from all scale rows to express the source's four physical planes.
        let make_row = |columns: usize, row: usize, role: usize| {
            let mut dense = Vec::new();
            let mut codes = Vec::new();
            let mut scales = Vec::new();
            for block in 0..columns / 64 {
                let scale = half::f16::from_f32(((row * 7 + block * 11 + role * 3) % 15 + 1) as f32 / 1024.0);
                scales.extend_from_slice(&scale.to_bits().to_le_bytes());
                let integers = (0..64)
                    .map(|col| ((row * 13 + block * 19 + col * 7 + col / 3 + role * 11) % 4) as u8)
                    .collect::<Vec<_>>();
                dense.extend(integers.iter().map(|&q| f64::from(scale.to_f32()) * (f64::from(q) - 1.0)));
                codes.extend(integers.chunks_exact(4).map(|q| q[0] | q[1] << 2 | q[2] << 4 | q[3] << 6));
            }
            (dense, codes, scales)
        };
        let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
        let (mut gu_codes, mut down_codes, mut gu_scales, mut down_scales) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in 0..intermediate {
            for role in 0..2 {
                let (dense, codes, scales) = make_row(hidden, row, role);
                if role == 0 {
                    gate.extend(dense);
                } else {
                    up.extend(dense);
                }
                gu_codes.extend(codes);
                gu_scales.extend(scales);
            }
        }
        for row in 0..hidden {
            let (dense, codes, scales) = make_row(intermediate, row, 2);
            down.extend(dense);
            down_codes.extend(codes);
            down_scales.extend(scales);
        }
        let bytes = [gu_codes, down_codes, gu_scales, down_scales].concat();
        Self {
            bytes,
            gate,
            up,
            down,
        }
    }
}

fn project(
    weights: &[f64],
    input: &[f64],
    columns: usize,
) -> (Vec<f64>, Vec<f64>) {
    let rows = weights.len() / columns;
    let (mut values, mut sums) = (Vec::new(), Vec::new());
    for token in input.chunks_exact(columns) {
        for row in weights.chunks_exact(columns) {
            let (mut sum, mut sumabs) = (0.0, 0.0);
            for (&weight, &x) in row.iter().zip(token) {
                let product = weight * x;
                sum += product;
                sumabs += product.abs();
            }
            values.push(sum);
            sums.push(sumabs);
        }
    }
    assert!(values.len() == rows * (input.len() / columns));
    (values, sums)
}

fn fused(
    gate: &[f64],
    up: &[f64],
) -> Vec<f64> {
    gate.iter().zip(up).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect()
}

fn check_matrix(
    actual: &[f32],
    expected: &[f64],
    sumabs: &[f64],
) {
    assert!(actual.len() == expected.len() && expected.len() == sumabs.len());
    for ((&actual, &expected), &sumabs) in actual.iter().zip(expected).zip(sumabs) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!((f64::from(actual) - expected).abs() <= 2e-5 * sumabs.max(1.0), "original-plane Q2 F64 matrix bound");
    }
}

fn check_fused(
    actual: &[f32],
    expected: &[f64],
) {
    assert!(actual.len() == expected.len());
    let (mut squared, mut norm, mut max) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        let delta = f64::from(actual) - expected;
        squared += delta * delta;
        norm += expected * expected;
        max = max.max(delta.abs());
    }
    assert!(
        max <= 0.002 && (squared / norm.max(f64::MIN_POSITIVE)).sqrt() <= 0.0002,
        "original-plane Q2 F64 fused bound"
    );
}

#[stark_test]
fn edge0_mtp_q2_original_planes_match_dense_f64_for_batches() {
    for (hidden, intermediate) in [(64, 64), (320, 192), (2560, 640)] {
        let fixture = Fixture::new(hidden, intermediate);
        // Odd offsets also exercise byte-based F16 access; guard bytes differ
        // from every valid scale and are outside the transport's exact span.
        let matrix = mapped(&fixture.bytes, 7);
        for tokens in [1usize, 2, 3, 8] {
            let input = (0..hidden * tokens)
                .map(|i| ((i * 17 + i / hidden * 23) % 53) as f32 * 0.037 - 0.83)
                .collect::<Vec<_>>();
            let latent = (0..intermediate * tokens)
                .map(|i| ((i * 19 + i / intermediate * 31) % 47) as f32 * 0.041 - 0.94)
                .collect::<Vec<_>>();
            let input64 = input.iter().copied().map(f64::from).collect::<Vec<_>>();
            let latent64 = latent.iter().copied().map(f64::from).collect::<Vec<_>>();
            let (gate, _) = project(&fixture.gate, &input64, hidden);
            let (up, _) = project(&fixture.up, &input64, hidden);
            let expected_fused = fused(&gate, &up);
            let (expected_down, sumabs) = project(&fixture.down, &latent64, intermediate);
            let (expected_composed, _) = project(&fixture.down, &expected_fused, intermediate);
            let mut graph = DecodeGraph::new().unwrap();
            let input = DecodeGraph::upload(&input).unwrap();
            let latent = DecodeGraph::upload(&latent).unwrap();
            let activated = graph.edge0_mtp_q2_gate_up(&matrix, hidden, intermediate, &input, tokens).unwrap();
            let down = graph.edge0_mtp_q2_down(&matrix, hidden, intermediate, &latent, tokens).unwrap();
            let composed = graph.edge0_mtp_q2_down(&matrix, hidden, intermediate, &activated, tokens).unwrap();
            let actual = graph.finish(&[activated, down, composed]).unwrap();
            check_fused(&actual[0], &expected_fused);
            check_matrix(&actual[1], &expected_down, &sumabs);
            check_fused(&actual[2], &expected_composed);
        }
    }
}

#[stark_test]
fn edge0_mtp_q2_first_last_rows_and_four_adjacent_code_lanes() {
    let (hidden, intermediate) = (128, 64);
    let mut fixture = Fixture::new(hidden, intermediate);
    // Eliminate accumulation error: single nonzero activation probes every
    // adjacent two-bit lane and both ends of every logical matrix.
    for column in [0usize, 1, 2, 3, 60, 63, 64, 127] {
        let mut input = vec![0.0_f32; hidden];
        input[column] = 1.0;
        let mut latent = vec![0.0_f32; intermediate];
        latent[column % intermediate] = 1.0;
        let mut graph = DecodeGraph::new().unwrap();
        let matrix = mapped(&fixture.bytes, 5);
        let x = DecodeGraph::upload(&input).unwrap();
        let z = DecodeGraph::upload(&latent).unwrap();
        let gu = graph.edge0_mtp_q2_gate_up(&matrix, hidden, intermediate, &x, 1).unwrap();
        let down = graph.edge0_mtp_q2_down(&matrix, hidden, intermediate, &z, 1).unwrap();
        let actual = graph.finish(&[gu, down]).unwrap();
        let gate = (0..intermediate).map(|row| fixture.gate[row * hidden + column]).collect::<Vec<_>>();
        let up = (0..intermediate).map(|row| fixture.up[row * hidden + column]).collect::<Vec<_>>();
        check_fused(&actual[0], &fused(&gate, &up));
        let expected =
            (0..hidden).map(|row| fixture.down[row * intermediate + column % intermediate]).collect::<Vec<_>>();
        check_matrix(&actual[1], &expected, &expected.iter().map(|v| v.abs()).collect::<Vec<_>>());
    }
    // Changing the final down-scale bytes must affect the final row. The
    // expected value is known independently of the packed-plane offsets.
    let last = fixture.bytes.len();
    fixture.bytes[last - 2..].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
    let matrix = mapped(&fixture.bytes, 0);
    let mut graph = DecodeGraph::new().unwrap();
    let input = DecodeGraph::upload(&vec![1.0; intermediate]).unwrap();
    let output = graph.edge0_mtp_q2_down(&matrix, hidden, intermediate, &input, 1).unwrap();
    let actual = graph.finish(&[output]).unwrap().remove(0);
    let row = &fixture.down[(hidden - 1) * intermediate..];
    let original_scale = (((hidden - 1) * 7 + 2 * 3) % 15 + 1) as f64 / 1024.0;
    let expected = row.iter().sum::<f64>() / original_scale * 0.5;
    assert!((f64::from(actual[hidden - 1]) - expected).abs() <= 2e-5 * expected.abs().max(1.0));
}

#[stark_test]
fn edge0_mtp_q2_rejects_malformed_geometry_and_transport() {
    let fixture = Fixture::new(64, 64);
    let matrix = mapped(&fixture.bytes, 3);
    let input = DecodeGraph::zeros(64).unwrap();
    let mut graph = DecodeGraph::new().unwrap();
    for (hidden, intermediate, tokens) in [
        (0, 64, 1),
        (64, 0, 1),
        (63, 64, 1),
        (64, 65, 1),
        (2624, 64, 1),
        (64, 704, 1),
        (usize::MAX, 64, 1),
        (64, 64, 0),
        (64, 64, 9),
        (64, 64, 2),
    ] {
        assert!(graph.edge0_mtp_q2_gate_up(&matrix, hidden, intermediate, &input, tokens).is_err());
        assert!(graph.edge0_mtp_q2_down(&matrix, hidden, intermediate, &input, tokens).is_err());
    }
    let mut malformed = Vec::new();
    let mut short = matrix.clone();
    short.codes.length -= 1;
    malformed.push(short);
    let mut long = matrix.clone();
    long.codes.length += 1;
    malformed.push(long);
    let mut truncated = matrix.clone();
    truncated.codes.registered_length = truncated.codes.offset + truncated.codes.length - 1;
    malformed.push(truncated);
    let mut unregistered = matrix.clone();
    unregistered.codes.registered_length = usize::MAX;
    malformed.push(unregistered);
    let mut overflow = matrix.clone();
    overflow.codes.offset = usize::MAX - 7;
    malformed.push(overflow);
    let mut planes = matrix.clone();
    planes.planes = Some((matrix.codes.clone(), matrix.codes.clone()));
    malformed.push(planes);
    for bad in malformed {
        assert!(graph.edge0_mtp_q2_gate_up(&bad, 64, 64, &input, 1).is_err());
        assert!(graph.edge0_mtp_q2_down(&bad, 64, 64, &input, 1).is_err());
    }
    let mut invalid = input.clone();
    invalid.offset = usize::MAX - 3;
    assert!(graph.edge0_mtp_q2_gate_up(&matrix, 64, 64, &invalid, 1).is_err());
    assert!(graph.edge0_mtp_q2_down(&matrix, 64, 64, &invalid, 1).is_err());
    graph.unretained = true;
    assert!(graph.edge0_mtp_q2_gate_up(&matrix, 64, 64, &input, 1).is_err());
    assert!(graph.edge0_mtp_q2_down(&matrix, 64, 64, &input, 1).is_err());
    graph.unretained = false;
    graph.projection_level = true;
    assert!(graph.edge0_mtp_q2_gate_up(&matrix, 64, 64, &input, 1).is_err());
    graph.projection_level = false;
    graph.finish(&[]).unwrap();
}

#[stark_test]
fn edge0_mtp_q2_nonfinite_weights_inputs_and_masked_intermediates_are_sticky() {
    let fixture = Fixture::new(64, 64);
    // For this geometry, GU code/down code/GU scale/down scale planes begin
    // at literal source-layout boundaries 0/2048/3072/3328, record size 3456.
    assert!(fixture.bytes.len() == 3456);
    for scale_offset in [3072usize, 3074, 3324, 3326, 3328, 3454] {
        for poison in [0x7c00_u16, 0xfc00, 0x7e00] {
            let mut bytes = fixture.bytes.clone();
            bytes[scale_offset..scale_offset + 2].copy_from_slice(&poison.to_le_bytes());
            let matrix = mapped(&bytes, 1);
            let mut graph = DecodeGraph::new().unwrap();
            let zeros = DecodeGraph::zeros(64).unwrap();
            let output = if scale_offset < 3328 {
                graph.edge0_mtp_q2_gate_up(&matrix, 64, 64, &zeros, 1).unwrap()
            } else {
                graph.edge0_mtp_q2_down(&matrix, 64, 64, &zeros, 1).unwrap()
            };
            // Finish an unrelated clean output too: earlier failure cannot be
            // cleared by a later valid projection or omitted return value.
            let clean = mapped(&fixture.bytes, 0);
            let clean_output = graph.edge0_mtp_q2_down(&clean, 64, 64, &zeros, 1).unwrap();
            assert!(graph.finish(&[output, clean_output]).is_err(), "nonfinite original-plane weight accepted");
        }
    }
    let matrix = mapped(&fixture.bytes, 0);
    for down in [false, true] {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0usize, 63] {
                let mut graph = DecodeGraph::new().unwrap();
                let input = DecodeGraph::zeros(64).unwrap();
                // Shared initialized storage bypasses upload's finite check.
                unsafe {
                    input.buffer.contents().as_ptr().cast::<f32>().add(index).write(poison);
                }
                let output = if down {
                    graph.edge0_mtp_q2_down(&matrix, 64, 64, &input, 1).unwrap()
                } else {
                    graph.edge0_mtp_q2_gate_up(&matrix, 64, 64, &input, 1).unwrap()
                };
                assert!(graph.finish(&[output]).is_err(), "nonfinite Q2 activation accepted");
            }
        }
    }
    // Finite inputs and finite weights produce a nonfinite up reduction.
    // Gate is exactly zero, which can mask this through fast-math rewriting;
    // the preactivation's explicit finite check must still set sticky failure.
    let mut bytes = vec![0x55_u8; 3456];
    for scale in bytes[3072..].chunks_exact_mut(2) {
        scale.copy_from_slice(&half::f16::from_f32(65504.0).to_bits().to_le_bytes());
    }
    for row in 0..64 {
        bytes[(row * 2 + 1) * 16..(row * 2 + 2) * 16].fill(0xff);
    }
    let matrix = mapped(&bytes, 0);
    let mut graph = DecodeGraph::new().unwrap();
    let input = DecodeGraph::upload(&[f32::MAX; 64]).unwrap();
    let output = graph.edge0_mtp_q2_gate_up(&matrix, 64, 64, &input, 1).unwrap();
    assert!(graph.finish(&[output]).is_err(), "masked nonfinite gate/up intermediate accepted");
}
