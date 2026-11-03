// SPDX-License-Identifier: Apache-2.0
//! Test-only F32 few-row MMA for the v39 Q5/Q6 projections.
//!
//! Scheduling/lane mapping and host admission derive from MIT llama.cpp
//! dde879bc4f954b809892fd0d660515f42536394a: kernels/mul_mv_mma.metal
//! (SHA256 115e843e70a6b5f8b6d4dc691a14706fadf6e9929b7f661989b5d7ae3d7716f9),
//! ggml-metal-common.cpp:46-140 and ggml-metal-device.cpp:846-960.
//! Stored weights and byte-safe dequantization remain Stark's. F32 MMA/K-slice
//! reductions change association and are checked against both v39 Safe kernels
//! and F64 products of both source-decoded F32 and mathematical GGUF weights.
//! The runtime wrapper has separate actual-shape/device admission gates below;
//! no activation downcast is introduced. The Q5 half-fragment specialization is
//! intentionally excluded from this initial matching-precision experiment.

use stark_engine_macros::stark_test;

use super::*;

#[derive(Clone, Copy)]
struct Shape {
    kind: u32,
    rows: usize,
    columns: usize,
    tokens: usize,
}

impl Shape {
    fn sizes(self) -> Result<(usize, usize)> {
        if !matches!(self.kind, 13 | 14)
            || self.rows == 0
            || self.columns == 0
            || !self.columns.is_multiple_of(256)
            || !(2..=16).contains(&self.tokens)
            || self.tokens.checked_mul(self.columns).is_none_or(|n| n > 8_000_000)
            || self.tokens.checked_mul(self.rows).is_none_or(|n| n > 8_000_000)
        {
            return Err(Error::Invalid("few-row MMA shape admission"));
        }
        let (_, block_bytes) = weights::block_layout(self.kind)?;
        let row_bytes = self.columns / 256 * block_bytes;
        let bytes = self.rows.checked_mul(row_bytes).ok_or(Error::Invalid("few-row MMA weight size overflow"))?;
        Ok((row_bytes, bytes))
    }
}

#[derive(Clone, Copy)]
struct MmaGeometry {
    nsg: usize,
    nt: usize,
    rt: usize,
    shared_bytes: usize,
}

fn halve_to_limit(
    mut value: usize,
    limit: usize,
) -> usize {
    while value > 1 && value > limit {
        value /= 2;
    }
    value
}

fn initial_geometry(shape: Shape) -> Result<MmaGeometry> {
    shape.sizes()?;
    let rt = shape.tokens.div_ceil(8);
    let nsg = halve_to_limit(
        if shape.rows <= 64 {
            32
        } else if shape.rows <= 6144 {
            16
        } else {
            8
        },
        shape.columns / 64,
    );
    let nt = halve_to_limit(4, (16_384 / (nsg * rt * 256)).min(shape.rows / (128 * 8)));
    Ok(MmaGeometry {
        nsg,
        nt,
        rt,
        shared_bytes: nsg * nt * rt * 256,
    })
}

fn mma_pipeline(shape: Shape) -> Result<(PipelineState, MmaGeometry)> {
    let executor = executor()?;
    let mut geometry = initial_geometry(shape)?;
    loop {
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in
            [(0, shape.kind), (4, 256), (5, weights::block_layout(shape.kind)?.1 as u32), (80, geometry.nsg as u32)]
        {
            constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
        }
        let name = format!("hybrid_kquant_mma_nt{}_rt{}", geometry.nt, geometry.rt);
        let function = executor
            .library
            .new_function_with_name_constant_values(&name, &constants)
            .map_err(|_| Error::Unsupported("few-row MMA function unavailable"))?;
        let pipeline = executor
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| Error::Unsupported("few-row MMA pipeline unavailable"))?;
        if pipeline.thread_execution_width() != 32 {
            return Err(Error::Unsupported("few-row MMA requires the qualified Apple 32-lane layout"));
        }
        if pipeline.max_total_threads_per_threadgroup() < geometry.nsg * 32 {
            if geometry.nsg == 1 {
                return Err(Error::Unsupported("few-row MMA pipeline thread limit"));
            }
            geometry.nsg /= 2;
            geometry.shared_bytes = geometry.nsg * geometry.nt * geometry.rt * 256;
            continue;
        }
        if geometry.shared_bytes + pipeline.static_threadgroup_memory_length()
            > executor.device.max_threadgroup_memory_length()
        {
            return Err(Error::Unsupported("few-row MMA threadgroup memory limit"));
        }
        return Ok((pipeline, geometry));
    }
}

fn serial_pipeline(kind: u32) -> Result<PipelineState> {
    let executor = executor()?;
    match kind {
        13 => executor.cached_pipeline(
            MatrixFormat::Packed(kind),
            "hybrid_q5_grouped32_matvec",
            &executor.q5_grouped32_pipelines,
        ),
        14 => executor.cached_pipeline(
            MatrixFormat::Packed(kind),
            "hybrid_q6_grouped16_matvec",
            &executor.q6_grouped16_pipelines,
        ),
        _ => Err(Error::Invalid("few-row MMA source format")),
    }
}

fn validate_views(
    shape: Shape,
    codes: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    offsets: [usize; 3],
) -> Result<()> {
    let (_, code_bytes) = shape.sizes()?;
    for (offset, bytes, length) in [
        (offsets[0], code_bytes, codes.length()),
        (
            offsets[1].checked_mul(4).ok_or(Error::Invalid("MMA input offset overflow"))?,
            shape.tokens * shape.columns * 4,
            input.length(),
        ),
        (
            offsets[2].checked_mul(4).ok_or(Error::Invalid("MMA output offset overflow"))?,
            shape.tokens * shape.rows * 4,
            output.length(),
        ),
    ] {
        if offset.checked_add(bytes).is_none_or(|end| end > length) {
            return Err(Error::Invalid("few-row MMA buffer span admission"));
        }
    }
    if offsets.iter().any(|&offset| offset > u32::MAX as usize) {
        return Err(Error::Invalid("few-row MMA parameter offset admission"));
    }
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "explicit component buffers and checked view offsets")]
fn encode_mma(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    shape: Shape,
    pipeline: &PipelineState,
    geometry: MmaGeometry,
    codes: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    failure: &ProtocolObject<dyn MTLBuffer>,
    offsets: [usize; 3],
) -> Result<()> {
    validate_views(shape, codes, input, output, offsets)?;
    if failure.length() < 4
        || geometry.rt != shape.tokens.div_ceil(8)
        || ![1, 2, 4].contains(&geometry.nt)
        || geometry.nsg == 0
        || geometry.nsg * 32 > pipeline.max_total_threads_per_threadgroup()
    {
        return Err(Error::Invalid("few-row MMA dispatch admission"));
    }
    encoder.set_compute_pipeline_state(pipeline);
    for (index, value) in [codes, input, output].into_iter().enumerate() {
        encoder.set_buffer(Some(value), 0, index);
    }
    let p = [
        shape.rows as u32,
        shape.columns as u32,
        shape.tokens as u32,
        offsets[0] as u32,
        offsets[1] as u32,
        offsets[2] as u32,
    ];
    encoder.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
    encoder.set_buffer(Some(failure), 0, 4);
    encoder.set_threadgroup_memory_length(geometry.shared_bytes, 0);
    encoder.dispatch_threadgroups(
        MTLSize {
            width: shape.rows.div_ceil(8 * geometry.nt),
            height: shape.tokens.div_ceil(8 * geometry.rt),
            depth: 1,
        },
        MTLSize {
            width: 32 * geometry.nsg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "same buffers as the few-row component with one source token")]
fn encode_serial(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    shape: Shape,
    pipeline: &PipelineState,
    codes: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    offsets: [usize; 3],
    token: usize,
) -> Result<()> {
    validate_views(shape, codes, input, output, offsets)?;
    if token >= shape.tokens {
        return Err(Error::Invalid("few-row source token admission"));
    }
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(Some(codes), 0, 0);
    encoder.set_buffer(Some(input), (offsets[1] + token * shape.columns) * 4, 1);
    encoder.set_buffer(Some(output), (offsets[2] + token * shape.rows) * 4, 2);
    let p = [
        shape.rows as u32,
        shape.columns as u32,
        shape.kind,
        256,
        weights::block_layout(shape.kind)?.1 as u32,
        0,
        0,
        0,
    ];
    let packed_offsets = [offsets[0] as u32, 0, 0, 0];
    encoder.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
    encoder.set_bytes(NonNull::from(&packed_offsets).cast(), std::mem::size_of_val(&packed_offsets), 6);
    encoder.dispatch_threadgroups(
        MTLSize {
            width: shape.rows.div_ceil(if shape.kind == 13 {
                2
            } else {
                4
            }),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn finite_codes(
    kind: u32,
    rows: usize,
    columns: usize,
    seed: usize,
) -> Vec<u8> {
    let bytes = weights::block_layout(kind).unwrap().1;
    let mut state = 0x9135_acedu32 ^ seed as u32;
    let mut codes = vec![0u8; rows * columns / 256 * bytes];
    for byte in &mut codes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    for (index, block) in codes.chunks_exact_mut(bytes).enumerate() {
        let d = half::f16::from_f32(0.001 + ((index * 17 + seed) % 257) as f32 * 0.00003125).to_bits().to_le_bytes();
        if kind == 14 {
            block[208..210].copy_from_slice(&d);
        } else {
            block[..2].copy_from_slice(&d);
            block[2..4].copy_from_slice(
                &half::f16::from_f32(0.0004 + ((index * 29 + seed) % 131) as f32 * 0.000015625).to_bits().to_le_bytes(),
            );
        }
    }
    codes
}

fn mathematical_weight(
    kind: u32,
    block: &[u8],
    index: usize,
) -> f64 {
    let half_at = |at: usize| f64::from(half::f16::from_bits(u16::from_le_bytes([block[at], block[at + 1]])).to_f32());
    if kind == 14 {
        let n = index / 128;
        let part = index % 128 / 32;
        let lane = index % 32;
        let low = (block[n * 64 + (part % 2) * 32 + lane]
            >> if part >= 2 {
                4
            } else {
                0
            })
            & 15;
        let high = (block[128 + n * 32 + lane] >> (part * 2)) & 3;
        let q = i32::from(low | (high << 4)) - 32;
        let scale = block[192 + n * 8 + lane / 16 + part * 2] as i8;
        return half_at(208) * f64::from(scale) * f64::from(q);
    }
    let part = index / 64;
    let high_nibble = index % 64 / 32;
    let lane = index % 32;
    let group = part * 2 + high_nibble;
    let scales = &block[4..16];
    let (scale, minimum) = if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    };
    let low = (block[48 + part * 32 + lane] >> (high_nibble * 4)) & 15;
    let q = low | (((block[16 + lane] >> group) & 1) << 4);
    half_at(0) * f64::from(scale) * f64::from(q) - half_at(2) * f64::from(minimum)
}

fn numerical_errors(
    shape: Shape,
    codes: &[u8],
    input: &[f32],
    actual: &[f32],
    source: &[f32],
    sample_rows: usize,
) -> (f64, f64, f64) {
    assert!(actual.len() == source.len() && actual.len() == shape.rows * shape.tokens, "MMA result shape");
    let scale = source.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    let squared = actual.iter().zip(source).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum::<f64>();
    let max_absolute =
        actual.iter().zip(source).map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0, f64::max);
    let row_bytes = shape.columns / 256 * weights::block_layout(shape.kind).unwrap().1;
    let samples = sample_rows.min(shape.rows).max(1);
    let mut normalized = 0.0f64;
    for token in 0..shape.tokens {
        for sample in 0..samples {
            let row = if samples == 1 {
                0
            } else {
                sample * (shape.rows - 1) / (samples - 1)
            };
            let mut expected = 0.0f64;
            let mut absolute = 0.0f64;
            let mut mathematical = 0.0f64;
            let mut mathematical_absolute = 0.0f64;
            for (encoded, values) in codes[row * row_bytes..(row + 1) * row_bytes]
                .chunks_exact(weights::block_layout(shape.kind).unwrap().1)
                .zip(input[token * shape.columns..(token + 1) * shape.columns].chunks_exact(256))
            {
                let mut decoded = [0.0f32; 256];
                weights::decode_block(shape.kind, encoded, &mut decoded).unwrap();
                for (index, (weight, value)) in decoded.into_iter().zip(values).enumerate() {
                    let product = f64::from(weight) * f64::from(*value);
                    expected += product;
                    absolute += product.abs();
                    let product = mathematical_weight(shape.kind, encoded, index) * f64::from(*value);
                    mathematical += product;
                    mathematical_absolute += product.abs();
                }
            }
            normalized =
                normalized.max((f64::from(actual[token * shape.rows + row]) - expected).abs() / absolute.max(1.0));
            normalized = normalized.max(
                (f64::from(actual[token * shape.rows + row]) - mathematical).abs() / mathematical_absolute.max(1.0),
            );
        }
    }
    ((squared / scale.max(f64::MIN_POSITIVE)).sqrt(), normalized, max_absolute)
}

fn run(
    shape: Shape,
    codes: &[u8],
    input: &[f32],
    mma: bool,
) -> Result<Vec<f32>> {
    shape.sizes()?;
    if codes.len() != shape.sizes()?.1 || input.len() != shape.tokens * shape.columns {
        return Err(Error::Invalid("few-row test payload shape"));
    }
    objc2::rc::autoreleasepool(|_| {
        let executor = executor()?;
        let mut storage = vec![0xa7u8; 19];
        storage.extend_from_slice(codes);
        storage.extend_from_slice(&[0x4d; 19]);
        let codes = buffer(executor, &storage)?;
        let mut values = vec![f32::NAN; 3];
        values.extend_from_slice(input);
        values.extend_from_slice(&[f32::NAN; 3]);
        let input = buffer(executor, bytemuck::cast_slice(&values))?;
        let count = shape.rows * shape.tokens;
        let output = buffer(executor, bytemuck::cast_slice(&vec![7.25f32; count + 6]))?;
        let failure = buffer(executor, bytemuck::cast_slice(&[0u32]))?;
        let command = executor.queue.command_buffer().ok_or(Error::Invalid("MMA test command"))?;
        let encoder = command.compute_command_encoder().ok_or(Error::Invalid("MMA test encoder"))?;
        if mma {
            let (pipeline, geometry) = mma_pipeline(shape)?;
            encode_mma(&encoder, shape, &pipeline, geometry, &codes, &input, &output, &failure, [19, 3, 3])?;
        } else {
            let pipeline = serial_pipeline(shape.kind)?;
            for token in 0..shape.tokens {
                encode_serial(&encoder, shape, &pipeline, &codes, &input, &output, [19, 3, 3], token)?;
            }
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("MMA test execution"));
        }
        let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count + 6) };
        assert!(values[..3].iter().chain(&values[count + 3..]).all(|&v| v == 7.25), "MMA output tail sentinel changed");
        if unsafe { failure.contents().as_ptr().cast::<u32>().read() } != 0
            || values[3..count + 3].iter().any(|x| !x.is_finite())
        {
            return Err(Error::Invalid("nonfinite few-row MMA result"));
        }
        Ok(values[3..count + 3].to_vec())
    })
}

#[stark_test]
fn kquant_mma_geometry_matches_pinned_admission_without_runtime_changes() {
    for tokens in [2, 4, 8, 14] {
        let small = initial_geometry(Shape {
            kind: 14,
            rows: 4096,
            columns: 4096,
            tokens,
        })
        .unwrap();
        assert_eq!(
            (small.nsg, small.nt, small.rt, small.shared_bytes),
            if tokens <= 8 {
                (16, 4, 1, 16384)
            } else {
                (16, 2, 2, 16384)
            }
        );
        let large = initial_geometry(Shape {
            kind: 13,
            rows: 8192,
            columns: 4096,
            tokens,
        })
        .unwrap();
        assert_eq!((large.nsg, large.nt, large.rt), (8, 4, tokens.div_ceil(8)));
    }
    for shape in [
        Shape {
            kind: 12,
            rows: 13,
            columns: 256,
            tokens: 2,
        },
        Shape {
            kind: 14,
            rows: 0,
            columns: 256,
            tokens: 2,
        },
        Shape {
            kind: 14,
            rows: 13,
            columns: 255,
            tokens: 2,
        },
        Shape {
            kind: 13,
            rows: 13,
            columns: 256,
            tokens: 17,
        },
    ] {
        assert!(initial_geometry(shape).is_err());
    }
}

#[stark_test]
fn kquant_mma_basis_offsets_tails_and_f64_source_oracles() {
    for kind in [14u32, 13] {
        for tokens in [2usize, 4, 8, 14] {
            let shape = Shape {
                kind,
                rows: 13,
                columns: 768,
                tokens,
            };
            let codes = finite_codes(kind, shape.rows, shape.columns, 7);
            for basis in [true, false] {
                let indices = [0, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 767];
                let mut input = (0..shape.columns * tokens)
                    .map(|i| {
                        if basis {
                            0.0
                        } else {
                            (i as f32 * 0.017).sin()
                        }
                    })
                    .collect::<Vec<_>>();
                if basis {
                    for token in 0..tokens {
                        input[token * shape.columns + indices[token % indices.len()]] = 1.0;
                    }
                }
                let source = run(shape, &codes, &input, false).unwrap();
                let actual = run(shape, &codes, &input, true).unwrap();
                let (rms, f64_error, _) = numerical_errors(shape, &codes, &input, &actual, &source, shape.rows);
                assert!(rms <= 2e-5, "few-row MMA source RMS changed");
                assert!(f64_error <= 2e-6, "few-row MMA F64 product error");
            }
        }
        // Exercise clamped weight rows and masked stores with every NT, not
        // only the NT1 geometry selected by the small 13-row basis fixture.
        for rows in [2051usize, 4099] {
            let shape = Shape {
                kind,
                rows,
                columns: 256,
                tokens: 14,
            };
            let codes = finite_codes(kind, rows, shape.columns, 31);
            let input = (0..shape.columns * shape.tokens).map(|i| (i as f32 * 0.013).cos()).collect::<Vec<_>>();
            let source = run(shape, &codes, &input, false).unwrap();
            let actual = run(shape, &codes, &input, true).unwrap();
            let (rms, f64_error, _) = numerical_errors(shape, &codes, &input, &actual, &source, 16);
            assert!(rms <= 2e-5 && f64_error <= 2e-6, "few-row MMA NT2/NT4 row-tail error");
        }
    }
}

#[stark_test]
fn kquant_mma_preserves_f32_activation_detail_and_extreme_fallback() {
    for kind in [14u32, 13] {
        let shape = Shape {
            kind,
            rows: 13,
            columns: 256,
            tokens: 14,
        };
        let codes = finite_codes(kind, shape.rows, shape.columns, 11);
        let input = (0..shape.columns * shape.tokens)
            .map(|i| 1.0 + ((i % 5 + 1) as f32) * 2.0f32.powi(-14))
            .collect::<Vec<_>>();
        let source = run(shape, &codes, &input, false).unwrap();
        let actual = run(shape, &codes, &input, true).unwrap();
        assert!(
            numerical_errors(shape, &codes, &input, &actual, &source, shape.rows).0 <= 2e-5,
            "MMA lost F32 activation detail"
        );
        let half_input = input.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect::<Vec<_>>();
        let downcast = run(shape, &codes, &half_input, true).unwrap();
        assert!(
            actual.iter().zip(downcast).any(|(a, b)| a.to_bits() != b.to_bits()),
            "MMA activation precision probe is insensitive"
        );
        for extreme in [1e30, -1e30, f32::MAX] {
            let input = (0..shape.tokens * shape.columns)
                .map(|i| {
                    if i % 2 == 0 {
                        extreme
                    } else {
                        -extreme
                    }
                })
                .collect::<Vec<_>>();
            let source = run(shape, &codes, &input, false);
            let actual = run(shape, &codes, &input, true);
            assert!(source.is_err() == actual.is_err(), "MMA extreme failure contract changed");
            if let (Ok(source), Ok(actual)) = (source, actual) {
                assert!(
                    actual.iter().zip(source).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "MMA fallback row chronology changed"
                );
            }
        }
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut input = vec![0.0; shape.tokens * shape.columns];
            input[13 * shape.columns + 255] = poison;
            assert!(run(shape, &codes, &input, true).is_err(), "MMA concealed nonfinite activation");
        }
        let bytes = weights::block_layout(kind).unwrap().1;
        let mut zero = vec![0u8; bytes];
        if kind == 14 {
            zero[128..192].fill(0xaa);
            zero[192..208].fill(1);
            zero[208..210].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        } else {
            zero[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
            zero[4..16].fill(1);
        }
        let values = run(shape, &zero.repeat(shape.rows), &vec![f32::MAX; shape.tokens * shape.columns], true).unwrap();
        assert!(values.iter().all(|&x| x == 0.0), "MMA zero times MAX fallback changed");
    }
}

#[stark_test]
fn kquant_mma_runtime_graph_actual_shapes_source_f64_and_failure_gates() {
    for (kind, tokens) in [(14, 4), (13, 8), (14, 8)] {
        for (rows, columns) in [(8192, 4096), (4096, 8192)] {
            let shape = Shape {
                kind,
                rows,
                columns,
                tokens,
            };
            let codes = finite_codes(kind, rows, columns, rows + columns);
            let input = (0..tokens * columns).map(|i| ((i * 31 % 149) as f32 - 74.0) * 0.0078125).collect::<Vec<_>>();
            let source = run(shape, &codes, &input, false).unwrap();
            let mut encoded = vec![0xa7; 19];
            encoded.extend_from_slice(&codes);
            encoded.extend_from_slice(&[0xa7; 19]);
            let matrix = MappedMatrix {
                codes: MappedSlice {
                    buffer: buffer(executor().unwrap(), &encoded).unwrap(),
                    offset: 19,
                    length: codes.len(),
                    registered_length: encoded.len(),
                },
                planes: None,
            };
            let operation = MappedOperation {
                matrices: vec![matrix],
                inputs: &[],
                rows,
                columns,
                rows_per_batch: 0,
                format: MatrixFormat::Packed(kind),
            };
            let counter = if kind == 13 {
                "native.glm.prefill.q5_mma"
            } else {
                "native.glm.prefill.q6_mma"
            };
            let was_enabled = profile::enabled();
            profile::enable(true);
            let before = profile::snapshot().get(counter).map_or(0, |c| c.calls);
            let mut graph = DecodeGraph::new().unwrap();
            let mut padded = vec![f32::NAN; 3];
            padded.extend_from_slice(&input);
            padded.extend_from_slice(&[f32::NAN; 3]);
            let values = DeviceVector {
                buffer: buffer(executor().unwrap(), bytemuck::cast_slice(&padded)).unwrap(),
                offset: 12,
                len: input.len(),
            };
            let output = graph.mapped_same_weight(&operation, &values, tokens, true).unwrap();
            let actual = graph.finish(&[output]).unwrap().remove(0);
            let after = profile::snapshot().get(counter).map_or(0, |c| c.calls);
            profile::enable(was_enabled);
            let qualified = executor().unwrap().device.name() == "Apple M5 Max";
            assert_eq!(after - before, u64::from(qualified), "actual route device admission mismatch");
            let (rrms, normalized, _) = numerical_errors(shape, &codes, &input, &actual, &source, 16);
            assert!(rrms <= 2e-5 && normalized <= 2e-6, "runtime MMA source/F64 tolerance failed");

            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut bad = input.clone();
                bad[(tokens - 1) * columns + 255] = poison;
                let mut graph = DecodeGraph::new().unwrap();
                let values = DeviceVector {
                    buffer: buffer(executor().unwrap(), bytemuck::cast_slice(&bad)).unwrap(),
                    offset: 0,
                    len: bad.len(),
                };
                let output = graph.mapped_same_weight(&operation, &values, tokens, true).unwrap();
                assert!(graph.finish(&[output]).is_err(), "runtime MMA concealed nonfinite input");
            }
            let large = vec![1.0e21f32; tokens * columns];
            let expected = run(shape, &codes, &large, false).unwrap();
            let mut graph = DecodeGraph::new().unwrap();
            let values = graph.upload_into(&large).unwrap();
            let output = graph.mapped_same_weight(&operation, &values, tokens, true).unwrap();
            let actual = graph.finish(&[output]).unwrap().remove(0);
            assert!(
                actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                "runtime MMA extreme fallback changed original chronology"
            );

            let mut zero = vec![0u8; weights::block_layout(kind).unwrap().1];
            if kind == 14 {
                zero[128..192].fill(0xaa);
                zero[192..208].fill(1);
                zero[208..210].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
            } else {
                zero[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
                zero[4..16].fill(1);
            }
            let zero = zero.repeat(rows * columns / 256);
            let zero_matrix = MappedMatrix {
                codes: MappedSlice {
                    buffer: buffer(executor().unwrap(), &zero).unwrap(),
                    offset: 0,
                    length: zero.len(),
                    registered_length: zero.len(),
                },
                planes: None,
            };
            let mut graph = DecodeGraph::new().unwrap();
            let extreme = graph.upload_into(&vec![f32::MAX; tokens * columns]).unwrap();
            let output = graph
                .mapped_same_weight(
                    &MappedOperation {
                        matrices: vec![zero_matrix],
                        inputs: &[],
                        rows,
                        columns,
                        rows_per_batch: 0,
                        format: MatrixFormat::Packed(kind),
                    },
                    &extreme,
                    tokens,
                    true,
                )
                .unwrap();
            let actual = graph.finish(&[output]).unwrap().remove(0);
            assert!(actual.iter().all(|&value| value == 0.0), "runtime MMA zero times MAX fallback failed");

            for tail in (1..tokens).filter(|&tail| !(kind == 14 && tail == 4)) {
                let input = DecodeGraph::upload(&input[..tail * columns]).unwrap();
                let mut control = DecodeGraph::new().unwrap();
                let expected = control.mapped_same_weight(&operation, &input, tail, false).unwrap();
                let expected = control.finish(&[expected]).unwrap().remove(0);
                let mut fallback = DecodeGraph::new().unwrap();
                let actual = fallback.mapped_same_weight(&operation, &input, tail, true).unwrap();
                let actual = fallback.finish(&[actual]).unwrap().remove(0);
                assert!(
                    actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "tail route changed legacy bits"
                );
            }
            let mut graph = DecodeGraph::new().unwrap();
            assert!(graph.mapped_same_weight(&operation, &extreme, 0, true).is_err());
            assert!(graph.mapped_same_weight(&operation, &extreme, 9, true).is_err());
            assert!(graph.mapped_same_weight(&operation, &extreme, 3, true).is_err());
        }
    }
}
