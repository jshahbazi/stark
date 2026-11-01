// SPDX-License-Identifier: Apache-2.0
//! Safe test-only aligned F16 scale reads. Production defaults use byte reads.
use stark_engine_macros::stark_test;

use super::*;

fn pipeline(
    kind: u32,
    typed: bool,
    code_offset: usize,
) -> Result<PipelineState> {
    if !matches!(kind, 12 | 13 | 14) || (typed && !code_offset.is_multiple_of(2)) {
        return Err(Error::Invalid("typed half diagnostic requires even code offset"));
    }
    let e = executor()?;
    let (_, bytes) = weights::block_layout(kind)?;
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, bytes as u32), (73, 0)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let stem = match kind {
        12 => "hybrid_q4_grouped32",
        13 => "hybrid_q5_grouped32",
        _ => "hybrid_q6_grouped16",
    };
    let name = if typed {
        format!("{stem}_typedhalf_matvec")
    } else {
        format!("{stem}_matvec")
    };
    let function = e
        .library
        .new_function_with_name_constant_values(&name, &constants)
        .map_err(|_| Error::Invalid("typed half diagnostic Safe function"))?;
    e.device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|_| Error::Invalid("typed half diagnostic Safe pipeline"))
}

fn codes(
    kind: u32,
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let (_, bytes) = weights::block_layout(kind).unwrap();
    let mut state = 0x9135_acedu32;
    let mut codes = vec![0u8; rows * columns / 256 * bytes];
    for byte in &mut codes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    for (i, block) in codes.chunks_exact_mut(bytes).enumerate() {
        let d = half::f16::from_f32(0.001 + ((i * 17) % 257) as f32 * 0.00003125).to_bits().to_le_bytes();
        if kind == 14 {
            block[208..210].copy_from_slice(&d);
        } else {
            block[..2].copy_from_slice(&d);
            block[2..4].copy_from_slice(
                &half::f16::from_f32(0.0004 + ((i * 29) % 131) as f32 * 0.000015625).to_bits().to_le_bytes(),
            );
        }
    }
    codes
}

fn rows_per_group(kind: u32) -> usize {
    if matches!(kind, 12 | 13) {
        2
    } else {
        4
    }
}

fn run(
    pipeline: &PipelineState,
    codes: &[u8],
    input: &[f32],
    parameters: [u32; 8],
    rows_per_group: usize,
    code_offset: usize,
) -> Result<Vec<f32>> {
    objc2::rc::autoreleasepool(|_| {
        let executor = executor()?;
        let count = parameters[0] as usize;
        let mut padded = vec![0x75u8; code_offset];
        padded.extend_from_slice(codes);
        let codes = buffer(executor, &padded)?;
        let input = buffer(executor, bytemuck::cast_slice(input))?;
        let output = buffer(executor, bytemuck::cast_slice(&vec![7.25f32; count + 2]))?;
        let command = executor.queue.command_buffer().unwrap();
        let encoding = command.compute_command_encoder().unwrap();
        encoding.set_compute_pipeline_state(pipeline);
        for (index, value) in [&codes, &input, &output].into_iter().enumerate() {
            encoding.set_buffer(
                Some(value),
                match index {
                    0 => code_offset,
                    2 => 4,
                    _ => 0,
                },
                index,
            );
        }
        encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
        let offsets = [0u32; 4];
        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
        encoding.dispatch_threadgroups(
            MTLSize {
                width: count.div_ceil(rows_per_group),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoding.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: the owned shared output has a completed GPU writer.
        let values = unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), count + 2) };
        assert_eq!(values[0], 7.25, "typed half scale prefix guard changed");
        assert_eq!(values[count + 1], 7.25, "typed half scale row tail guard changed");
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("nonfinite typed half scale test result"));
        }
        Ok(values[1..count + 1].to_vec())
    })
}

fn f64_oracle(
    kind: u32,
    codes: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
    rows_per_batch: usize,
) -> Vec<(f64, f64)> {
    let (block, bytes) = weights::block_layout(kind).unwrap();
    let row_bytes = columns / block * bytes;
    (0..rows)
        .map(|row| {
            let input_start = if rows_per_batch == 0 {
                0
            } else {
                row / rows_per_batch * columns
            };
            let mut expected = 0.0f64;
            let mut absolute = 0.0f64;
            for (encoded, input) in codes[row * row_bytes..(row + 1) * row_bytes]
                .chunks_exact(bytes)
                .zip(input[input_start..input_start + columns].chunks_exact(block))
            {
                let mut decoded = vec![0.0; block];
                weights::decode_block(kind, encoded, &mut decoded).unwrap();
                for (weight, input) in decoded.iter().zip(input) {
                    let product = f64::from(*weight) * f64::from(*input);
                    expected += product;
                    absolute += product.abs();
                }
            }
            (expected, absolute)
        })
        .collect()
}

#[stark_test]
fn kquant_typed_half_preserves_bits_f64_subnormals_offsets_and_fallbacks() {
    for kind in [12u32, 13, 14] {
        let (_, bytes) = weights::block_layout(kind).unwrap();
        assert!(pipeline(kind, true, 1).is_err(), "typed half accepted odd mapped offset");
        for columns in [256usize, 768, 4096, 8192] {
            let rows = 39usize;
            let p = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, 13, 0];
            let mut encoded = codes(kind, rows, columns);
            // Include both signs of zero, subnormals, min normal and max F16.
            let finite = [0x0000u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
            for (i, block) in encoded.chunks_exact_mut(bytes).enumerate() {
                let at = if kind == 14 {
                    208
                } else {
                    0
                };
                block[at..at + 2].copy_from_slice(&finite[i % finite.len()].to_le_bytes());
                if kind != 14 {
                    block[2..4].copy_from_slice(&finite[(i + 3) % finite.len()].to_le_bytes());
                }
            }
            let input = (0..columns * 3).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let wide = f64_oracle(kind, &encoded, rows, columns, &input, 13);
            if columns == 256 {
                let byte = pipeline(kind, false, 0).unwrap();
                let expected = run(&byte, &encoded, &input, p, rows_per_group(kind), 0).unwrap();
                let actual = run(&byte, &encoded, &input, p, rows_per_group(kind), 1).unwrap();
                assert!(
                    actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "byte fallback changed bits at odd mapped offset"
                );
            }
            for offset in [0usize, 2] {
                let byte = pipeline(kind, false, offset).unwrap();
                let typed = pipeline(kind, true, offset).unwrap();
                let reference = run(&byte, &encoded, &input, p, rows_per_group(kind), offset).unwrap();
                let actual = run(&typed, &encoded, &input, p, rows_per_group(kind), offset).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "typed half scale changed Safe output bits"
                );
                for (row, &(expected, absolute)) in wide.iter().enumerate() {
                    assert!(
                        (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                        "typed half scale F64 gate"
                    );
                }
                let large = (0..columns * 3)
                    .map(|i| {
                        if i % 2 == 0 {
                            1e21
                        } else {
                            -1e21
                        }
                    })
                    .collect::<Vec<_>>();
                let a = run(&byte, &encoded, &large, p, rows_per_group(kind), offset);
                let b = run(&typed, &encoded, &large, p, rows_per_group(kind), offset);
                assert_eq!(a.is_err(), b.is_err(), "typed half extreme failure mismatch");
                if let (Ok(a), Ok(b)) = (a, b) {
                    assert!(
                        a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "typed half exact extreme fallback"
                    );
                }
                for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                    let mut invalid = vec![0.0; columns * 3];
                    invalid[columns - 1] = poison;
                    let a = run(&byte, &encoded, &invalid, p, rows_per_group(kind), offset);
                    let b = run(&typed, &encoded, &invalid, p, rows_per_group(kind), offset);
                    assert_eq!(a.is_err(), b.is_err(), "typed half input failure parity");
                    if let (Ok(a), Ok(b)) = (a, b) {
                        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
                    }
                }
            }
        }
        let (rows, columns) = (13usize, 256usize);
        let p = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, 0, 0];
        let byte = pipeline(kind, false, 2).unwrap();
        let typed = pipeline(kind, true, 2).unwrap();
        for bits in [0x7c00u16, 0xfc00, 0x7e00, 0xfe00] {
            for at in if kind == 14 {
                vec![208usize]
            } else {
                vec![0usize, 2]
            } {
                let mut encoded = codes(kind, rows, columns);
                encoded[at..at + 2].copy_from_slice(&bits.to_le_bytes());
                let input = vec![1.0; columns];
                assert!(run(&byte, &encoded, &input, p, rows_per_group(kind), 2).is_err());
                assert!(
                    run(&typed, &encoded, &input, p, rows_per_group(kind), 2).is_err(),
                    "typed half nonfinite scale escaped"
                );
            }
        }
        let zero = vec![0u8; rows * bytes];
        let maximum = vec![f32::MAX; columns];
        let a = run(&byte, &zero, &maximum, p, rows_per_group(kind), 2).unwrap();
        let b = run(&typed, &zero, &maximum, p, rows_per_group(kind), 2).unwrap();
        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()), "typed half zero times MAX contract");
        assert!(run(&typed, &codes(kind, rows, columns), &vec![1.0; columns], p, rows_per_group(kind), 2).is_ok());
    }
}

#[stark_test]
#[ignore]
fn kquant_typed_half_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let cases = [
        ("glm_q4k_readout", 12u32, 154880usize, 4096usize),
        ("glm_q5k_main", 13u32, 8192usize, 4096usize),
        ("glm_q5k_output", 13, 4096, 8192),
        ("glm_q6k_main", 14, 8192, 4096),
        ("glm_q6k_output", 14, 4096, 8192),
        ("glm_q6k_shared", 14, 4096, 2048),
    ];
    let executor = executor().unwrap();
    for (label, kind, rows, columns) in cases {
        let pipelines = [
            ("byte_scales", 0u32, pipeline(kind, false, 0).unwrap()),
            ("typed_half_scales", 1, pipeline(kind, true, 0).unwrap()),
        ];
        let (_, block_bytes) = weights::block_layout(kind).unwrap();
        let matrix_bytes = rows * columns / 256 * block_bytes;
        let row_bytes = matrix_bytes / rows;
        let banks = (160usize * 1024 * 1024).div_ceil(matrix_bytes);
        let codes = codes(kind, rows * banks, columns);
        let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let mut oracle = Vec::new();
        for bank in 0..banks {
            for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                let at = bank * matrix_bytes + row * row_bytes;
                let expected = f64_oracle(kind, &codes[at..at + row_bytes], 1, columns, &input, 0)[0];
                oracle.push((bank * rows + row, expected));
            }
        }
        let codes = buffer(executor, &codes).unwrap();
        let input = buffer(executor, bytemuck::cast_slice(&input)).unwrap();
        let output = executor.device.new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let parameters = [rows as u32, columns as u32, kind, 256, block_bytes as u32, 0, 0, 0];
        let mut timings = vec![vec![Vec::<f64>::new(); 2]; 1];
        let mut reference = Vec::<f32>::new();
        let mut bitwise_equal = [true; 2];
        let mut maximum_absolute_error = [0.0f64; 2];
        let mut relative_rms_error = [0.0f64; 2];
        let mut maximum_f64_error_over_absolute_sum = [0.0f64; 2];
        for replay in 0..7 {
            for candidate in [1usize] {
                for variant in [0, candidate, candidate, 0] {
                    let (_, _, pipeline) = &pipelines[variant];
                    let (values, seconds) = objc2::rc::autoreleasepool(|_| {
                        let command = executor.queue.command_buffer().unwrap();
                        let encoding = command.compute_command_encoder().unwrap();
                        encoding.set_compute_pipeline_state(pipeline);
                        encoding.set_buffer(Some(&input), 0, 1);
                        encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
                        let offsets = [0u32; 4];
                        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
                        for i in 0..banks {
                            let bank = (i + replay) % banks;
                            encoding.set_buffer(Some(&codes), bank * matrix_bytes, 0);
                            encoding.set_buffer(Some(&output), bank * rows * 4, 2);
                            encoding.dispatch_threadgroups(
                                MTLSize {
                                    width: rows.div_ceil(rows_per_group(kind)),
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: 64,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        }
                        encoding.end_encoding();
                        command.commit();
                        command.wait_until_completed();
                        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                        // SAFETY: all owned shared output banks have completed writers.
                        let values = unsafe {
                            std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * banks)
                        }
                        .to_vec();
                        (values, (command.gpu_end_time() - command.gpu_start_time()) / banks as f64)
                    });
                    assert!(values.iter().all(|v| v.is_finite()), "typed half scale component nonfinite");
                    if reference.is_empty() {
                        reference = values.clone();
                    }
                    bitwise_equal[variant] &= values.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits());
                    if replay > 0 {
                        timings[candidate - 1][usize::from(variant != 0)].push(seconds);
                    }
                    maximum_absolute_error[variant] = maximum_absolute_error[variant].max(
                        values
                            .iter()
                            .zip(&reference)
                            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                            .fold(0.0, f64::max),
                    );
                    let squared_error: f64 =
                        values.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    relative_rms_error[variant] = relative_rms_error[variant]
                        .max((squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt());
                    for &(row, (expected, absolute)) in &oracle {
                        maximum_f64_error_over_absolute_sum[variant] = maximum_f64_error_over_absolute_sum[variant]
                            .max((f64::from(values[row]) - expected).abs() / absolute.max(1.0));
                    }
                }
            }
        }
        for (pair, schedules) in timings.iter_mut().enumerate() {
            for (which, samples) in schedules.iter_mut().enumerate() {
                samples.sort_by(f64::total_cmp);
                let seconds = (samples[5] + samples[6]) * 0.5;
                let variant = if which == 0 {
                    0
                } else {
                    pair + 1
                };
                let (schedule, _, pipeline) = &pipelines[variant];
                println!(
                    "{}",
                    serde_json::json!({
                        "component": label, "kind": kind, "rows": rows, "columns": columns,
                        "math_mode": "safe", "typed_half_scales":variant==1, "effective_code_offset":0, "schedule": schedule, "activity_assertion": true,
                        "pair_candidate": pipelines[pair+1].0,
                        "rows_per_group": rows_per_group(kind), "threads_per_group": 64,
                        "pipeline_max_threads": pipeline.max_total_threads_per_threadgroup(),
                        "rotating_banks": banks, "encoded_population_bytes": matrix_bytes * banks,
                        "warm_samples": samples.len(), "median_gpu_microseconds_per_matrix": seconds * 1e6,
                        "minimum_gpu_microseconds_per_matrix": samples[0] * 1e6,
                        "maximum_gpu_microseconds_per_matrix": samples[samples.len() - 1] * 1e6,
                        "encoded_gigabytes_per_second": matrix_bytes as f64 / seconds / 1e9,
                        "maximum_absolute_error_vs_original": maximum_absolute_error[variant],
                        "relative_rms_error_vs_original": relative_rms_error[variant], "bitwise_equal_to_original":bitwise_equal[variant],
                        "maximum_f64_error_over_absolute_sum": maximum_f64_error_over_absolute_sum[variant],
                        "f64_oracle_rows": oracle.len(),
                        "timer_scope": "same_process_safe_ABBA_serial_typed_half_scale_reads",
                    })
                );
                assert!(bitwise_equal[variant], "typed half component changed Safe arithmetic bits");
                assert!(relative_rms_error[variant] <= 2e-5, "typed half scale component source relative RMS");
                assert!(maximum_f64_error_over_absolute_sum[variant] <= 2e-6, "typed half scale component F64 gate");
            }
        }
    }
}
