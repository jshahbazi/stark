// SPDX-License-Identifier: Apache-2.0
//! Safe test-only Q8 schedules; source provenance is in the companion shader.
use stark_engine_macros::stark_test;

use super::*;

fn pipeline(variant: usize) -> PipelineState {
    let executor = executor().unwrap();
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, 8u32), (4, 32), (5, 34)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let name = match variant {
        0 => "hybrid_packed_block_matvec",
        1 => "hybrid_q8_grouped8_matvec",
        2 => "hybrid_q8_segmented8_matvec",
        4 => "hybrid_q8_grouped8_r4g2_matvec",
        5 => "hybrid_q8_segmented4_matvec",
        _ => "hybrid_packed_matvec",
    };
    let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}
fn geometry(variant: usize) -> (usize, usize) {
    match variant {
        0 => (2, 64),
        1 => (2, 128),
        2 => (8, 64),
        4 => (4, 64),
        5 => (16, 64),
        _ => (1, 256),
    }
}
fn codes(
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let mut state = 0x874b19d3u32;
    let mut bytes = vec![0u8; rows * columns / 32 * 34];
    for byte in &mut bytes {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *byte = (state >> 24) as u8;
    }
    for (i, block) in bytes.chunks_exact_mut(34).enumerate() {
        block[..2].copy_from_slice(
            &half::f16::from_f32(0.001 + ((i * 17) % 257) as f32 * 0.00003125).to_bits().to_le_bytes(),
        );
    }
    bytes
}
fn production_pipeline(variant: usize) -> PipelineState {
    let executor = executor().unwrap();
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, 8u32), (4, 32), (5, 34)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let name = if variant == 1 {
        "hybrid_q8_grouped8_safe_matvec"
    } else {
        "hybrid_q8_segmented8_safe_matvec"
    };
    let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

#[stark_test]
fn q8_production_wrappers_preserve_checked_schedule_and_output_failures() {
    for variant in [1usize, 2] {
        let checked = pipeline(variant);
        let production = production_pipeline(variant);
        let (r, t) = geometry(variant);
        for columns in [128usize, 1536] {
            let encoded = codes(13, columns);
            let p = [13, columns as u32, 8, 32, 34, 0, 0, 0];
            for magnitude in [1.0f32, 1e20, 1e30] {
                let input = (0..columns).map(|i| (i as f32 * 0.017).sin() * magnitude).collect::<Vec<_>>();
                let expected = run(&checked, &encoded, &input, p, r, t).unwrap();
                let actual = run(&production, &encoded, &input, p, r, t).unwrap();
                assert!(
                    actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "Q8 raw-compatible wrapper changed checked schedule bits"
                );
            }
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut input = vec![0.0; columns];
                input[columns - 1] = poison;
                assert!(
                    run(&production, &encoded, &input, p, r, t).is_err(),
                    "Q8 raw-compatible wrapper lost nonfinite result"
                );
            }
        }
    }
}

#[stark_test]
fn q8_grouped8_preserves_f64_tails_batches_and_exact_extreme_fallback() {
    let pipelines = [pipeline(0), pipeline(1), pipeline(2)];
    for columns in [32usize, 128, 1536, 4096] {
        for rows_per_batch in [0usize, 13] {
            let rows = if rows_per_batch == 0 {
                13
            } else {
                39
            };
            let encoded = codes(rows, columns);
            let p = [rows as u32, columns as u32, 8, 32, 34, 0, rows_per_batch as u32, 0];
            for magnitude in [1.0f32, 1e20] {
                let input = (0..columns
                    * if rows_per_batch == 0 {
                        1
                    } else {
                        3
                    })
                    .map(|i| (i as f32 * 0.017).sin() * magnitude)
                    .collect::<Vec<_>>();
                let reference = run(&pipelines[0], &encoded, &input, p, 2, 64).unwrap();
                let oracle = f64_oracle(8, &encoded, rows, columns, &input, rows_per_batch);
                for (variant, candidate) in pipelines.iter().enumerate() {
                    let (r, t) = geometry(variant);
                    let actual = run(candidate, &encoded, &input, p, r, t).unwrap();
                    for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "Q8 grouped F64 gate"
                        );
                    }
                    let error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let power: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!((error / power.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "Q8 grouped source RMS");
                }
            }
            let large = (0..columns
                * if rows_per_batch == 0 {
                    1
                } else {
                    3
                })
                .map(|i| {
                    if i % 2 == 0 {
                        1e30
                    } else {
                        -1e30
                    }
                })
                .collect::<Vec<_>>();
            let reference = run(&pipelines[0], &encoded, &large, p, 2, 64).unwrap();
            for (variant, candidate) in pipelines.iter().enumerate().skip(1) {
                let (r, t) = geometry(variant);
                let actual = run(candidate, &encoded, &large, p, r, t).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "Q8 extreme fallback bits changed"
                );
                for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                    let mut invalid = vec![0.0f32; large.len()];
                    invalid[columns - 1] = poison;
                    let expected = run(&pipelines[0], &encoded, &invalid, p, 2, 64);
                    let actual = run(candidate, &encoded, &invalid, p, r, t);
                    assert_eq!(actual.is_err(), expected.is_err(), "Q8 poison/overflow failure mismatch");
                    if let (Ok(a), Ok(b)) = (actual, expected) {
                        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
                    }
                }
                assert!(
                    run(candidate, &encoded, &vec![0.0; large.len()], p, r, t).is_ok(),
                    "Q8 fresh-command recovery"
                );
            }
        }
    }
    for poison in [half::f16::NAN, half::f16::INFINITY, half::f16::NEG_INFINITY] {
        let mut encoded = codes(13, 128);
        encoded[..2].copy_from_slice(&poison.to_bits().to_le_bytes());
        for (variant, candidate) in pipelines.iter().enumerate() {
            let (r, t) = geometry(variant);
            assert!(
                run(candidate, &encoded, &vec![1.0; 128], [13, 128, 8, 32, 34, 0, 0, 0], r, t).is_err(),
                "Q8 nonfinite block scale escaped"
            );
        }
    }
    let mut zero = vec![0u8; 34];
    zero[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
    for (variant, candidate) in pipelines.iter().enumerate() {
        let (r, t) = geometry(variant);
        let actual = run(candidate, &zero.repeat(13), &vec![f32::MAX; 32], [13, 32, 8, 32, 34, 0, 0, 0], r, t).unwrap();
        assert!(actual.iter().all(|v| v.to_bits() == 0), "Q8 zero times MAX lost legacy behavior");
    }
}

#[stark_test]
fn q8_followup_schedules_preserve_f64_tails_batches_and_exact_extreme_fallback() {
    let variants = [0usize, 4, 5];
    let pipelines = [pipeline(0), pipeline(4), pipeline(5)];
    for columns in [32usize, 128, 1536, 4096] {
        for rows_per_batch in [0usize, 13] {
            let rows = if rows_per_batch == 0 {
                13
            } else {
                39
            };
            let encoded = codes(rows, columns);
            let p = [rows as u32, columns as u32, 8, 32, 34, 0, rows_per_batch as u32, 0];
            for magnitude in [1.0f32, 1e20] {
                let input = (0..columns
                    * if rows_per_batch == 0 {
                        1
                    } else {
                        3
                    })
                    .map(|i| (i as f32 * 0.017).sin() * magnitude)
                    .collect::<Vec<_>>();
                let reference = run(&pipelines[0], &encoded, &input, p, 2, 64).unwrap();
                let oracle = f64_oracle(8, &encoded, rows, columns, &input, rows_per_batch);
                for (variant, candidate) in pipelines.iter().enumerate() {
                    let (r, t) = geometry(variants[variant]);
                    let actual = run(candidate, &encoded, &input, p, r, t).unwrap();
                    for (row, &(expected, absolute)) in oracle.iter().enumerate() {
                        assert!(
                            (f64::from(actual[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "Q8 grouped F64 gate"
                        );
                    }
                    let error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let power: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!((error / power.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "Q8 grouped source RMS");
                }
            }
            let large = (0..columns
                * if rows_per_batch == 0 {
                    1
                } else {
                    3
                })
                .map(|i| {
                    if i % 2 == 0 {
                        1e30
                    } else {
                        -1e30
                    }
                })
                .collect::<Vec<_>>();
            let reference = run(&pipelines[0], &encoded, &large, p, 2, 64).unwrap();
            for (variant, candidate) in pipelines.iter().enumerate().skip(1) {
                let (r, t) = geometry(variants[variant]);
                let actual = run(candidate, &encoded, &large, p, r, t).unwrap();
                assert!(
                    actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "Q8 extreme fallback bits changed"
                );
                for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                    let mut invalid = vec![0.0f32; large.len()];
                    invalid[columns - 1] = poison;
                    let expected = run(&pipelines[0], &encoded, &invalid, p, 2, 64);
                    let actual = run(candidate, &encoded, &invalid, p, r, t);
                    assert_eq!(actual.is_err(), expected.is_err(), "Q8 poison/overflow failure mismatch");
                    if let (Ok(a), Ok(b)) = (actual, expected) {
                        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
                    }
                }
                assert!(
                    run(candidate, &encoded, &vec![0.0; large.len()], p, r, t).is_ok(),
                    "Q8 fresh-command recovery"
                );
            }
        }
    }
    for poison in [half::f16::NAN, half::f16::INFINITY, half::f16::NEG_INFINITY] {
        let mut encoded = codes(13, 128);
        encoded[..2].copy_from_slice(&poison.to_bits().to_le_bytes());
        for (variant, candidate) in pipelines.iter().enumerate() {
            let (r, t) = geometry(variants[variant]);
            assert!(
                run(candidate, &encoded, &vec![1.0; 128], [13, 128, 8, 32, 34, 0, 0, 0], r, t).is_err(),
                "Q8 nonfinite block scale escaped"
            );
        }
    }
    let mut zero = vec![0u8; 34];
    zero[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
    for (variant, candidate) in pipelines.iter().enumerate() {
        let (r, t) = geometry(variants[variant]);
        let actual = run(candidate, &zero.repeat(13), &vec![f32::MAX; 32], [13, 32, 8, 32, 34, 0, 0, 0], r, t).unwrap();
        assert!(actual.iter().all(|v| v.to_bits() == 0), "Q8 zero times MAX lost legacy behavior");
    }
}

fn run(
    pipeline: &PipelineState,
    codes: &[u8],
    input: &[f32],
    parameters: [u32; 8],
    rows_per_group: usize,
    threads: usize,
) -> Result<Vec<f32>> {
    objc2::rc::autoreleasepool(|_| {
        let executor = executor()?;
        let count = parameters[0] as usize;
        let codes = buffer(executor, codes)?;
        let input = buffer(executor, bytemuck::cast_slice(input))?;
        let output = buffer(executor, bytemuck::cast_slice(&vec![7.25f32; count + 2]))?;
        let failure = buffer(executor, bytemuck::cast_slice(&[0u32]))?;
        let command = executor.queue.command_buffer().unwrap();
        let encoding = command.compute_command_encoder().unwrap();
        encoding.set_compute_pipeline_state(pipeline);
        encoding.set_buffer(Some(&failure), 0, 30);
        for (index, value) in [&codes, &input, &output].into_iter().enumerate() {
            encoding.set_buffer(
                Some(value),
                if index == 2 {
                    4
                } else {
                    0
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
                width: threads,
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
        assert_eq!(values[0], 7.25, "Q8 grouped prefix guard changed");
        assert_eq!(values[count + 1], 7.25, "Q8 grouped row tail guard changed");
        if unsafe { *failure.contents().as_ptr().cast::<u32>() } != 0 || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("nonfinite Q8 grouped test result"));
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
#[ignore]
fn q8_grouped8_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let cases = [
        ("glm_q8_mla_qb", 8u32, 16384usize, 1536usize),
        ("glm_q8_kda_fb", 8, 8192, 128),
        ("glm_q8_kda_fa_excluded", 8, 128, 4096),
    ];
    let executor = executor().unwrap();
    for (label, kind, rows, columns) in cases {
        let pipelines = [
            ("block2", 0usize, pipeline(0)),
            ("grouped8", 1, pipeline(1)),
            ("segmented8", 2, pipeline(2)),
            ("scalar_control", 3, pipeline(3)),
        ];
        let (_, block_bytes) = weights::block_layout(kind).unwrap();
        let matrix_bytes = rows * columns / 32 * block_bytes;
        let row_bytes = matrix_bytes / rows;
        let banks = (160usize * 1024 * 1024).div_ceil(matrix_bytes);
        let codes = codes(rows * banks, columns);
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
        let failure = buffer(executor, bytemuck::cast_slice(&[0u32])).unwrap();
        let parameters = [rows as u32, columns as u32, kind, 32, block_bytes as u32, 0, 0, 0];
        let mut timings = vec![vec![Vec::<f64>::new(); 2]; 3];
        let mut reference = Vec::<f32>::new();
        let mut maximum_absolute_error = [0.0f64; 4];
        let mut relative_rms_error = [0.0f64; 4];
        let mut maximum_f64_error_over_absolute_sum = [0.0f64; 4];
        for replay in 0..7 {
            for candidate in if replay % 2 == 0 {
                [1usize, 2, 3]
            } else {
                [3usize, 2, 1]
            } {
                for variant in [0, candidate, candidate, 0] {
                    let (_, _, pipeline) = &pipelines[variant];
                    let (row_group, threads) = geometry(variant);
                    let (values, seconds) = objc2::rc::autoreleasepool(|_| {
                        let command = executor.queue.command_buffer().unwrap();
                        let encoding = command.compute_command_encoder().unwrap();
                        encoding.set_compute_pipeline_state(pipeline);
                        encoding.set_buffer(Some(&failure), 0, 30);
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
                                    width: rows.div_ceil(row_group),
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
                    assert_eq!(unsafe { *failure.contents().as_ptr().cast::<u32>() }, 0, "Q8 component sticky failure");
                    assert!(values.iter().all(|v| v.is_finite()), "Q8 grouped component nonfinite");
                    if reference.is_empty() {
                        reference = values.clone();
                    }
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
                        "component": label, "production_admitted":false, "activity":true, "threads_per_group":geometry(variant).1, "kind": kind, "rows": rows, "columns": columns,
                        "math_mode": "safe", "schedule": schedule, "activity_assertion": true,
                        "pair_candidate": pipelines[pair+1].0,
                        "rows_per_group": geometry(variant).0,
                        "pipeline_max_threads": pipeline.max_total_threads_per_threadgroup(),
                        "rotating_banks": banks, "encoded_population_bytes": matrix_bytes * banks,
                        "warm_samples": samples.len(), "median_gpu_microseconds_per_matrix": seconds * 1e6,
                        "minimum_gpu_microseconds_per_matrix": samples[0] * 1e6,
                        "maximum_gpu_microseconds_per_matrix": samples[samples.len() - 1] * 1e6,
                        "encoded_gigabytes_per_second": matrix_bytes as f64 / seconds / 1e9,
                        "maximum_absolute_error_vs_original": maximum_absolute_error[variant],
                        "relative_rms_error_vs_original": relative_rms_error[variant],
                        "maximum_f64_error_over_absolute_sum": maximum_f64_error_over_absolute_sum[variant],
                        "f64_oracle_rows": oracle.len(),
                        "timer_scope": "same_process_safe_ABBA_serial_rotating_banks_gpu_timestamps",
                    })
                );
                assert!(relative_rms_error[variant] <= 2e-5, "Q8 grouped component source relative RMS");
                assert!(maximum_f64_error_over_absolute_sum[variant] <= 2e-6, "Q8 grouped component F64 gate");
            }
        }
    }
}

#[stark_test]
#[ignore]
fn q8_followup_schedules_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let cases = [
        ("glm_q8_mla_qb", 8u32, 16384usize, 1536usize),
        ("glm_q8_kda_fb", 8, 8192, 128),
        ("glm_q8_kda_fa_excluded", 8, 128, 4096),
    ];
    let executor = executor().unwrap();
    for (label, kind, rows, columns) in cases {
        let baseline = if rows == 16384 {
            1usize
        } else if rows == 8192 {
            2
        } else {
            3
        };
        let reference_pipeline = if baseline == 3 {
            pipeline(3)
        } else {
            production_pipeline(baseline)
        };
        let pipelines = [
            ("qualified_shape", baseline, reference_pipeline),
            ("grouped8_r4g2", 4, pipeline(4)),
            ("segmented4", 5, pipeline(5)),
        ];
        let (_, block_bytes) = weights::block_layout(kind).unwrap();
        let matrix_bytes = rows * columns / 32 * block_bytes;
        let row_bytes = matrix_bytes / rows;
        let banks = (160usize * 1024 * 1024).div_ceil(matrix_bytes);
        let codes = codes(rows * banks, columns);
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
        let failure = buffer(executor, bytemuck::cast_slice(&[0u32])).unwrap();
        let parameters = [rows as u32, columns as u32, kind, 32, block_bytes as u32, 0, 0, 0];
        let mut timings = vec![vec![Vec::<f64>::new(); 2]; 2];
        let mut reference = Vec::<f32>::new();
        let mut maximum_absolute_error = [0.0f64; 3];
        let mut relative_rms_error = [0.0f64; 3];
        let mut maximum_f64_error_over_absolute_sum = [0.0f64; 3];
        for replay in 0..7 {
            for candidate in if replay % 2 == 0 {
                [1usize, 2]
            } else {
                [2usize, 1]
            } {
                for variant in [0, candidate, candidate, 0] {
                    let (_, schedule_id, pipeline) = &pipelines[variant];
                    let (row_group, threads) = geometry(*schedule_id);
                    let (values, seconds) = objc2::rc::autoreleasepool(|_| {
                        let command = executor.queue.command_buffer().unwrap();
                        let encoding = command.compute_command_encoder().unwrap();
                        encoding.set_compute_pipeline_state(pipeline);
                        encoding.set_buffer(Some(&failure), 0, 30);
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
                                    width: rows.div_ceil(row_group),
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
                    assert_eq!(unsafe { *failure.contents().as_ptr().cast::<u32>() }, 0, "Q8 component sticky failure");
                    assert!(values.iter().all(|v| v.is_finite()), "Q8 grouped component nonfinite");
                    if reference.is_empty() {
                        reference = values.clone();
                    }
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
                        "component": label, "production_admitted":false, "activity":true, "threads_per_group":geometry(pipelines[variant].1).1, "kind": kind, "rows": rows, "columns": columns,
                        "math_mode": "safe", "schedule": schedule, "activity_assertion": true,
                        "pair_candidate": pipelines[pair+1].0,
                        "rows_per_group": geometry(pipelines[variant].1).0,
                        "pipeline_max_threads": pipeline.max_total_threads_per_threadgroup(),
                        "rotating_banks": banks, "encoded_population_bytes": matrix_bytes * banks,
                        "warm_samples": samples.len(), "median_gpu_microseconds_per_matrix": seconds * 1e6,
                        "minimum_gpu_microseconds_per_matrix": samples[0] * 1e6,
                        "maximum_gpu_microseconds_per_matrix": samples[samples.len() - 1] * 1e6,
                        "encoded_gigabytes_per_second": matrix_bytes as f64 / seconds / 1e9,
                        "maximum_absolute_error_vs_original": maximum_absolute_error[variant],
                        "relative_rms_error_vs_original": relative_rms_error[variant],
                        "maximum_f64_error_over_absolute_sum": maximum_f64_error_over_absolute_sum[variant],
                        "f64_oracle_rows": oracle.len(),
                        "timer_scope": "same_process_safe_ABBA_serial_q8_followup_rotating_banks",
                    })
                );
                assert!(relative_rms_error[variant] <= 2e-5, "Q8 grouped component source relative RMS");
                assert!(maximum_f64_error_over_absolute_sum[variant] <= 2e-6, "Q8 grouped component F64 gate");
            }
        }
    }
}

#[stark_test]
#[ignore]
fn q8_grouped8_production_mixed_shapes_cache_geometry_and_numerics() {
    let format = MatrixFormat::Packed(8);
    assert_eq!(q8_grouped8_shape(format, 16384, 1536), Some(true), "fresh process requires STARK_GLM53_Q8_GROUPED8=1");
    let executor = executor().unwrap();
    for (rows, columns, variant) in [
        (16384usize, 1536usize, 4usize),
        (8192, 128, 2),
        (128, 4096, 3),
        (8192, 128, 2),
        (16384, 1536, 4),
        (13, 128, 3),
    ] {
        let (r, t) = geometry(variant);
        assert_eq!(matrix_groups(format, rows, columns), rows.div_ceil(r));
        assert_eq!(matrix_threads(format, rows, columns), t);
        let encoded = codes(rows, columns);
        let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let p = [rows as u32, columns as u32, 8, 32, 34, 0, 0, 0];
        let production = executor.run_inner(&encoded, &input, p, None, TestMatrixSchedule::Production).unwrap();
        let expected = if variant == 3 {
            executor.run_inner(&encoded, &input, p, None, TestMatrixSchedule::Scalar).unwrap()
        } else {
            run(&pipeline(variant), &encoded, &input, p, r, t).unwrap()
        };
        assert!(
            production.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
            "production Q8 shape/cache does not match qualified Safe schedule"
        );
        let reference = run(&pipeline(0), &encoded, &input, p, 2, 64).unwrap();
        let error: f64 = production.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
        let power: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
        assert!((error / power.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "production Q8 source RMS");
        let row_bytes = columns / 32 * 34;
        for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
            let (expected, absolute) =
                f64_oracle(8, &encoded[row * row_bytes..(row + 1) * row_bytes], 1, columns, &input, 0)[0];
            assert!(
                (f64::from(production[row]) - expected).abs() <= 2e-6 * absolute.max(1.0),
                "production Q8 F64 gate"
            );
        }
        if variant != 3 {
            let large = (0..columns)
                .map(|i| {
                    if i % 2 == 0 {
                        1e30
                    } else {
                        -1e30
                    }
                })
                .collect::<Vec<_>>();
            let expected = run(&pipeline(0), &encoded, &large, p, 2, 64).unwrap();
            let actual = executor.run_inner(&encoded, &large, p, None, TestMatrixSchedule::Production).unwrap();
            assert!(
                actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                "production Q8 extreme fallback bits"
            );
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = input.clone();
                invalid[columns - 1] = poison;
                assert!(
                    executor.run_inner(&encoded, &invalid, p, None, TestMatrixSchedule::Production).is_err(),
                    "production Q8 nonfinite raw command escaped"
                );
            }
        }
    }
    assert!(executor.q8_grouped8_pipelines.lock().unwrap().contains_key(&format));
    assert!(executor.q8_segmented8_pipelines.lock().unwrap().contains_key(&format));
    assert_eq!(indexed_name(format), "hybrid_graph_packed_indexed");
    assert_eq!(indexed_groups(format, 16384), 16384);
    assert_eq!(indexed_threads(format), 256);
    assert_eq!(q8_grouped8_shape(format, 16384, 0), None);
}

fn graph_projection(
    encoded: &[u8],
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    graph_projection_policy(encoded, input, rows, columns, false)
}

fn graph_projection_policy(
    encoded: &[u8],
    input: &[f32],
    rows: usize,
    columns: usize,
    ripple_q8: bool,
) -> Result<Vec<f32>> {
    let matrix = MappedMatrix {
        codes: MappedSlice {
            buffer: buffer(executor()?, encoded)?,
            offset: 0,
            length: encoded.len(),
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
        format: MatrixFormat::Packed(8),
    };
    let input_vector = DecodeGraph::zeros(input.len())?;
    // Poison is deliberately written after checked allocation to exercise the
    // actual graph's input finite scanner rather than upload admission.
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr(),
            input_vector.buffer.contents().as_ptr().cast::<f32>(),
            input.len(),
        );
    }
    let mut graph = DecodeGraph::new()?;
    if ripple_q8 {
        graph.enable_ripple_dense_q8();
    }
    let output = graph.mapped(&operation, &input_vector)?;
    Ok(graph.finish(&[output])?.remove(0))
}

#[stark_test]
fn ripple_q8_policy_is_scoped_to_measured_dense_shapes() {
    for (rows, columns) in [(40960, 5120), (5120, 40960), (10240, 5120), (1024, 5120), (5120, 10240), (100352, 5120)] {
        assert!(ripple_dense_q8_shape(MatrixFormat::Packed(8), rows, columns));
        for excluded in [MatrixFormat::Packed(0), MatrixFormat::Packed(12), MatrixFormat::Packed(30)] {
            assert!(!ripple_dense_q8_shape(excluded, rows, columns));
        }
        assert!(!ripple_dense_q8_shape(MatrixFormat::Packed(8), rows, 0), "indexed geometry entered dense policy");
    }
    for (rows, columns) in [(13, 128), (1024, 4096), (100353, 5120), (5120, 5120), (16384, 1536)] {
        assert!(!ripple_dense_q8_shape(MatrixFormat::Packed(8), rows, columns));
    }
    let rows = 1024usize;
    let columns = 5120usize;
    let encoded = codes(rows, columns);
    let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let p = [rows as u32, columns as u32, 8, 32, 34, 0, 0, 0];
    let default = graph_projection(&encoded, &input, rows, columns).unwrap();
    let scalar = run(&pipeline(3), &encoded, &input, p, 1, 256).unwrap();
    assert!(default.iter().zip(&scalar).all(|(a, b)| a.to_bits() == b.to_bits()), "other graph default changed");
    let optimized = graph_projection_policy(&encoded, &input, rows, columns, true).unwrap();
    let expected = run(&production_pipeline(1), &encoded, &input, p, 2, 128).unwrap();
    assert!(
        optimized.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "Ripple graph schedule/grid mismatch"
    );
    let error: f64 = optimized.iter().zip(&scalar).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
    let power: f64 = scalar.iter().map(|&v| f64::from(v).powi(2)).sum();
    assert!((error / power.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5, "Ripple Q8 scalar RMS");
}

#[stark_test]
fn ripple_q8_graph_preserves_extreme_failures_and_recovery() {
    let rows = 1024usize;
    let columns = 5120usize;
    let encoded = codes(rows, columns);
    let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
    let p = [rows as u32, columns as u32, 8, 32, 34, 0, 0, 0];
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut invalid = input.clone();
        invalid[columns - 1] = poison;
        let expected = run(&pipeline(0), &encoded, &invalid, p, 2, 64);
        let actual = graph_projection_policy(&encoded, &invalid, rows, columns, true);
        assert_eq!(actual.is_err(), expected.is_err(), "Ripple Q8 graph finite contract");
        if let (Ok(a), Ok(b)) = (actual, expected) {
            assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()), "Ripple Q8 extreme fallback changed");
        }
    }
    let large: Vec<_> = (0..columns)
        .map(|i| {
            if i % 2 == 0 {
                1e30
            } else {
                -1e30
            }
        })
        .collect();
    let expected = run(&pipeline(0), &encoded, &large, p, 2, 64).unwrap();
    let actual = graph_projection_policy(&encoded, &large, rows, columns, true).unwrap();
    assert!(
        actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        "Ripple Q8 large finite fallback changed"
    );
    for poison in [half::f16::NAN, half::f16::INFINITY, half::f16::NEG_INFINITY] {
        let mut invalid = encoded.clone();
        invalid[..2].copy_from_slice(&poison.to_bits().to_le_bytes());
        assert!(
            graph_projection_policy(&invalid, &input, rows, columns, true).is_err(),
            "Ripple Q8 nonfinite scale escaped"
        );
    }
    let mut zeros = vec![0u8; encoded.len()];
    for block in zeros.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
    }
    let maximum = vec![f32::MAX; columns];
    let actual = graph_projection_policy(&zeros, &maximum, rows, columns, true).unwrap();
    assert!(actual.iter().all(|v| v.to_bits() == 0), "Ripple Q8 zero times MAX contract");
    assert!(graph_projection_policy(&encoded, &input, rows, columns, true).is_ok(), "Ripple Q8 fresh-command recovery");
}

/// Reproducible, aggregate-only dense Ripple component experiment. The
/// rotating encoded population exceeds the on-chip cache; no model data is
/// used. Every candidate is bracketed by the original scalar schedule.
#[stark_test]
#[ignore]
fn q8_ripple_dense_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let executor = executor().unwrap();
    let variants = [3usize, 0, 1, 2, 4, 5];
    let names = ["scalar_control", "block2", "grouped8_r2g4", "segmented8", "grouped8_r4g2", "segmented4"];
    let pipelines = variants.map(pipeline);
    for (label, rows, columns) in [
        ("ripple_ffn_gate_up", 40960usize, 5120usize),
        ("ripple_ffn_down", 5120, 40960),
        ("ripple_attention_q_gate", 10240, 5120),
        ("ripple_attention_k_v", 1024, 5120),
        ("ripple_attention_o", 5120, 10240),
        ("ripple_output", 100352, 5120),
    ] {
        let matrix_bytes = rows * columns / 32 * 34;
        let row_bytes = matrix_bytes / rows;
        let banks = (320usize * 1024 * 1024).div_ceil(matrix_bytes);
        let encoded = codes(rows * banks, columns);
        let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let mut oracle = Vec::new();
        for bank in 0..banks {
            for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                let at = bank * matrix_bytes + row * row_bytes;
                oracle.push((bank * rows + row, f64_oracle(8, &encoded[at..at + row_bytes], 1, columns, &input, 0)[0]));
            }
        }
        let encoded = buffer(executor, &encoded).unwrap();
        let input = buffer(executor, bytemuck::cast_slice(&input)).unwrap();
        let output = executor.device.new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let failure = buffer(executor, bytemuck::cast_slice(&[0u32])).unwrap();
        let parameters = [rows as u32, columns as u32, 8u32, 32, 34, 0, 0, 0];
        let mut timings = vec![vec![Vec::<f64>::new(); 2]; variants.len() - 1];
        let mut reference = Vec::<f32>::new();
        let mut maximum_absolute_error = [0.0f64; 6];
        let mut relative_rms_error = [0.0f64; 6];
        let mut maximum_f64_error_over_absolute_sum = [0.0f64; 6];
        for replay in 0..7 {
            let candidates: Vec<_> = if replay % 2 == 0 {
                (1..variants.len()).collect()
            } else {
                (1..variants.len()).rev().collect()
            };
            for candidate in candidates {
                for variant in [0, candidate, candidate, 0] {
                    let pipeline = &pipelines[variant];
                    let (row_group, threads) = geometry(variants[variant]);
                    let (values, seconds) = objc2::rc::autoreleasepool(|_| {
                        let command = executor.queue.command_buffer().unwrap();
                        let encoding = command.compute_command_encoder().unwrap();
                        encoding.set_compute_pipeline_state(pipeline);
                        encoding.set_buffer(Some(&failure), 0, 30);
                        encoding.set_buffer(Some(&input), 0, 1);
                        encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
                        let offsets = [0u32; 4];
                        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
                        for i in 0..banks {
                            let bank = (i + replay) % banks;
                            encoding.set_buffer(Some(&encoded), bank * matrix_bytes, 0);
                            encoding.set_buffer(Some(&output), bank * rows * 4, 2);
                            encoding.dispatch_threadgroups(
                                MTLSize {
                                    width: rows.div_ceil(row_group),
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
                        encoding.end_encoding();
                        command.commit();
                        command.wait_until_completed();
                        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                        // SAFETY: all output banks have completed GPU writers.
                        let values = unsafe {
                            std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * banks)
                        }
                        .to_vec();
                        (values, (command.gpu_end_time() - command.gpu_start_time()) / banks as f64)
                    });
                    assert_eq!(
                        unsafe { *failure.contents().as_ptr().cast::<u32>() },
                        0,
                        "Ripple Q8 component sticky failure"
                    );
                    assert!(values.iter().all(|v| v.is_finite()), "Ripple Q8 component nonfinite");
                    if reference.is_empty() {
                        reference = values.clone();
                    }
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
                println!(
                    "{}",
                    serde_json::json!({
                        "component": label, "production_admitted": false, "rows": rows, "columns": columns, "kind": 8,
                        "math_mode": "safe", "schedule": names[variant], "pair_candidate": names[pair + 1],
                        "activity_assertion": true, "threads_per_group": geometry(variants[variant]).1,
                        "rows_per_group": geometry(variants[variant]).0, "rotating_banks": banks,
                        "encoded_population_bytes": matrix_bytes * banks, "warm_samples": samples.len(),
                        "median_gpu_microseconds_per_matrix": seconds * 1e6,
                        "minimum_gpu_microseconds_per_matrix": samples[0] * 1e6,
                        "maximum_gpu_microseconds_per_matrix": samples[samples.len() - 1] * 1e6,
                        "encoded_gigabytes_per_second": matrix_bytes as f64 / seconds / 1e9,
                        "maximum_absolute_error_vs_original": maximum_absolute_error[variant],
                        "relative_rms_error_vs_original": relative_rms_error[variant],
                        "maximum_f64_error_over_absolute_sum": maximum_f64_error_over_absolute_sum[variant],
                        "f64_oracle_rows": oracle.len(),
                        "timer_scope": "same_process_safe_ABBA_serial_ripple_rotating_banks_gpu_timestamps"
                    })
                );
                assert!(relative_rms_error[variant] <= 2e-5, "Ripple Q8 component source RMS");
                assert!(maximum_f64_error_over_absolute_sum[variant] <= 2e-6, "Ripple Q8 component F64 gate");
            }
        }
    }
}

#[stark_test]
#[ignore]
fn q8_grouped8_actual_raw_and_graph_preserve_failure_contracts() {
    assert_eq!(
        q8_grouped8_shape(MatrixFormat::Packed(8), 16384, 1536),
        Some(true),
        "fresh process requires STARK_GLM53_Q8_GROUPED8=1"
    );
    let executor = executor().unwrap();
    for (rows, columns) in [(16384usize, 1536usize), (8192, 128)] {
        let encoded = codes(rows, columns);
        let input = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let p = [rows as u32, columns as u32, 8, 32, 34, 0, 0, 0];
        let expected = executor.run_inner(&encoded, &input, p, None, TestMatrixSchedule::Production).unwrap();
        let actual = graph_projection(&encoded, &input, rows, columns).unwrap();
        assert!(
            actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
            "actual Q8 raw/graph normal mismatch"
        );
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut invalid = input.clone();
            invalid[columns - 1] = poison;
            let reference = executor.run_inner(&encoded, &invalid, p, None, TestMatrixSchedule::Block);
            let raw = executor.run_inner(&encoded, &invalid, p, None, TestMatrixSchedule::Production);
            let graph = graph_projection(&encoded, &invalid, rows, columns);
            assert_eq!(raw.is_err(), reference.is_err(), "Q8 actual raw input failure parity");
            assert_eq!(graph.is_err(), reference.is_err(), "Q8 actual graph input failure parity");
            if let (Ok(raw), Ok(graph), Ok(reference)) = (raw, graph, reference) {
                assert!(raw.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(graph.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
        for poison in [half::f16::NAN, half::f16::INFINITY, half::f16::NEG_INFINITY] {
            let mut invalid = encoded.clone();
            invalid[..2].copy_from_slice(&poison.to_bits().to_le_bytes());
            assert!(executor.run_inner(&invalid, &input, p, None, TestMatrixSchedule::Block).is_err());
            assert!(
                executor.run_inner(&invalid, &input, p, None, TestMatrixSchedule::Production).is_err(),
                "Q8 actual raw nonfinite scale escaped"
            );
            assert!(
                graph_projection(&invalid, &input, rows, columns).is_err(),
                "Q8 actual graph nonfinite scale escaped"
            );
        }
        let mut zeros = vec![0u8; encoded.len()];
        for block in zeros.chunks_exact_mut(34) {
            block[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        }
        let maximum = vec![f32::MAX; columns];
        let raw = executor.run_inner(&zeros, &maximum, p, None, TestMatrixSchedule::Production).unwrap();
        let graph = graph_projection(&zeros, &maximum, rows, columns).unwrap();
        assert!(raw.iter().chain(&graph).all(|v| v.to_bits() == 0), "Q8 actual zero times MAX contract");
        assert!(graph_projection(&encoded, &input, rows, columns).is_ok(), "Q8 actual graph fresh-command recovery");
    }
}
