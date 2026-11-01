// SPDX-License-Identifier: Apache-2.0
//! Constant-table experiment; production selection is unchanged.
use stark_engine_macros::stark_test;

use super::{tests::*, *};

fn pair(kind: u32) -> [PipelineState; 2] {
    [
        safe_iq_test_pipeline(kind, iq_bound_factor_name(MatrixFormat::Packed(kind)).unwrap(), 4),
        safe_iq_test_pipeline(
            kind,
            if kind == 17 {
                "hybrid_graph_iq_constant_iq2xs"
            } else {
                "hybrid_graph_iq_constant_iq3xxs"
            },
            4,
        ),
    ]
}

#[stark_test]
fn iq_constant_lookup_matches_f64_and_preserves_fallback_failures() {
    let (experts, rows) = (4usize, 13usize);
    let ids = [3u32, 1, 3];
    for kind in [17u32, 18] {
        let pipelines = pair(kind);
        for columns in [256usize, 2048, 4096] {
            let codes = finite_packed_codes(kind, experts * rows, columns, 7);
            let matrix_bytes = codes.len() / experts;
            let selected_codes = ids
                .iter()
                .flat_map(|&id| codes[id as usize * matrix_bytes..(id as usize + 1) * matrix_bytes].iter().copied())
                .collect::<Vec<_>>();
            for batched in [false, true] {
                let p = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
                let count = columns
                    * if batched {
                        ids.len()
                    } else {
                        1
                    };
                for magnitude in [1.0f32, 1e20, 1e30] {
                    let input = (0..count).map(|i| (i as f32 * 0.019).sin() * magnitude).collect::<Vec<_>>();
                    let run = |pipeline| {
                        run_iq_indexed_test_with_bound(pipeline, &codes, &input, &ids, p, 8, 64, true).unwrap()
                    };
                    let reference = run(&pipelines[0]);
                    let actual = run(&pipelines[1]);
                    let oracle = packed_f64_oracle(
                        kind,
                        &selected_codes,
                        rows * ids.len(),
                        columns,
                        &input,
                        if batched {
                            rows
                        } else {
                            0
                        },
                    );
                    for (&actual, (expected, absolute)) in actual.iter().zip(oracle) {
                        assert!(
                            (f64::from(actual) - expected).abs() <= 2e-6 * absolute.max(1.0),
                            "constant IQ F64 gate"
                        );
                    }
                    if magnitude > 1e20 {
                        assert!(
                            actual.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "constant IQ fallback changed"
                        );
                    }
                    let error: f64 =
                        actual.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let scale: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    assert!((error / scale.max(f64::MIN_POSITIVE)).sqrt() <= 2e-5);
                }
                for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                    let input = vec![poison; count];
                    assert!(
                        run_iq_indexed_test_with_bound(&pipelines[1], &codes, &input, &ids, p, 8, 64, true).is_err()
                    );
                }
                let normal = vec![0.25f32; count];
                assert!(
                    run_iq_indexed_test_with_bound(&pipelines[1], &codes, &normal, &[4, 1, 3], p, 8, 64, true).is_err()
                );
                assert!(run_iq_indexed_test_with_bound(&pipelines[1], &codes, &normal, &ids, p, 8, 64, true).is_ok());
            }
        }
    }
}

#[stark_test]
#[ignore]
fn iq_constant_lookup_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let cases = [("glm_iq2xs_gate", 17, 2048usize, 4096usize, false), ("glm_iq3xxs_down", 18, 4096, 2048, true)];
    let (experts, selected) = (64usize, 8usize);
    let groups = experts / selected;
    let executor = executor().unwrap();
    for (label, kind, rows, columns, batched) in cases {
        let (_, bytes) = weights::block_layout(kind).unwrap();
        let matrix_bytes = rows * columns / 256 * bytes;
        let mut codes = vec![0u8; matrix_bytes * experts];
        let mut state = 0x1b7a_95e3u32;
        for byte in &mut codes {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        for encoded in codes.chunks_exact_mut(bytes) {
            encoded[..2].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
        }
        let input = (0..columns
            * if batched {
                experts
            } else {
                1
            })
            .map(|i| (i as f32 * 0.017).sin())
            .collect::<Vec<_>>();
        let codes = buffer(executor, &codes).unwrap();
        let input = buffer(executor, bytemuck::cast_slice(&input)).unwrap();
        let indices = buffer(executor, bytemuck::cast_slice(&(0..experts as u32).collect::<Vec<_>>())).unwrap();
        let output = executor.device.new_buffer(rows * experts * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let failure = buffer(executor, bytemuck::cast_slice(&[0u32])).unwrap();
        let parameters = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
        let bounds = buffer(executor, bytemuck::cast_slice(&vec![0u32; selected])).unwrap();
        let function = executor.library.new_function_with_name("hybrid_graph_iq_input_bound").unwrap();
        let prepass = executor.device.new_compute_pipeline_state_with_function(&function).unwrap();
        let [original, candidate] = pair(kind);
        let pipelines = [
            ("factor_threadgroup", 4usize, 8usize, 64usize, original),
            ("factor_constant", 4usize, 8usize, 64usize, candidate),
        ];
        let mut timings = vec![vec![Vec::<f64>::new(); 2]; pipelines.len() - 1];
        let mut reference = Vec::<f32>::new();
        let mut bitwise_equal = [true; 4];
        let mut maximum_absolute_error = [0.0f64; 4];
        let mut relative_rms_error = [0.0f64; 4];
        for replay in 0..7 {
            for candidate in if replay % 2 == 0 {
                [1usize]
            } else {
                [1usize]
            } {
                for variant in [0, candidate, candidate, 0] {
                    let (_, _, rows_per_group, threads, pipeline) = &pipelines[variant];
                    let (values, seconds) = objc2::rc::autoreleasepool(|_| {
                        let command = executor.queue.command_buffer().unwrap();
                        let encoding = command.compute_command_encoder().unwrap();
                        encoding.set_compute_pipeline_state(pipeline);
                        encoding.set_buffer(Some(&codes), 0, 0);
                        encoding.set_buffer(Some(&failure), 0, 30);
                        encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
                        for group in 0..groups {
                            {
                                encoding.set_compute_pipeline_state(&prepass);
                                encoding.set_buffer(
                                    Some(&input),
                                    if batched {
                                        group * selected * columns * 4
                                    } else {
                                        0
                                    },
                                    0,
                                );
                                encoding.set_buffer(Some(&bounds), 0, 1);
                                let bound_parameters = [
                                    columns as u32,
                                    if batched {
                                        selected as u32
                                    } else {
                                        1
                                    },
                                ];
                                encoding.set_bytes(
                                    NonNull::from(&bound_parameters).cast(),
                                    std::mem::size_of_val(&bound_parameters),
                                    2,
                                );
                                encoding.dispatch_threadgroups(
                                    MTLSize {
                                        width: bound_parameters[1] as usize,
                                        height: 1,
                                        depth: 1,
                                    },
                                    MTLSize {
                                        width: 256,
                                        height: 1,
                                        depth: 1,
                                    },
                                );
                                encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
                                encoding.set_compute_pipeline_state(pipeline);
                                encoding.set_buffer(Some(&codes), 0, 0);
                                encoding.set_buffer(Some(&bounds), 0, 7);
                            }
                            encoding.set_buffer(
                                Some(&input),
                                if batched {
                                    group * selected * columns * 4
                                } else {
                                    0
                                },
                                1,
                            );
                            encoding.set_buffer(Some(&output), group * selected * rows * 4, 2);
                            encoding.set_buffer(Some(&indices), group * selected * 4, 6);
                            encoding.dispatch_threadgroups(
                                MTLSize {
                                    width: rows.div_ceil(*rows_per_group),
                                    height: selected,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: *threads,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        }
                        encoding.end_encoding();
                        command.commit();
                        command.wait_until_completed();
                        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                        // SAFETY: owned shared buffers have completed GPU
                        // writers covering all rows*experts output slots.
                        let values = unsafe {
                            std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * experts)
                        }
                        .to_vec();
                        assert_eq!(unsafe { *failure.contents().as_ptr().cast::<u32>() }, 0);
                        (values, (command.gpu_end_time() - command.gpu_start_time()) / groups as f64)
                    });
                    assert!(values.iter().all(|v| v.is_finite()), "IQ grouped32 component nonfinite");
                    if reference.is_empty() {
                        reference = values.clone();
                    }
                    if replay > 0 {
                        timings[candidate - 1][usize::from(variant != 0)].push(seconds);
                    }
                    bitwise_equal[variant] &= values.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits());
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
                let (schedule, rows_per_simd, rows_per_group, threads, pipeline) = &pipelines[variant];
                println!(
                    "{}",
                    serde_json::json!({
                        "component": label, "kind": kind, "rows": rows, "columns": columns,
                        "math_mode": "safe", "gpu_prepass_included":true, "pso_hint":false, "signed_factor":true, "constant_lookup":variant == 1, "activity":true, "host_verified_finite_bound":1.0e20, "pair_candidate_schedule": pipelines[pair + 1].0,
                        "schedule": schedule, "rows_per_simd": rows_per_simd, "rows_per_group": rows_per_group,
                        "threads_per_group": threads, "pipeline_max_threads": pipeline.max_total_threads_per_threadgroup(),
                        "selected_experts": selected, "experts": experts, "rotating_groups": groups,
                        "encoded_population_bytes": matrix_bytes * experts, "warm_samples": samples.len(),
                        "median_gpu_microseconds_per_8_experts": seconds * 1e6,
                        "minimum_gpu_microseconds_per_8_experts": samples[0] * 1e6,
                        "maximum_gpu_microseconds_per_8_experts": samples[samples.len() - 1] * 1e6,
                        "encoded_gigabytes_per_second": (matrix_bytes * selected) as f64 / seconds / 1e9,
                        "bitwise_equal_to_original": bitwise_equal[variant],
                        "maximum_absolute_error_vs_original": maximum_absolute_error[variant],
                        "relative_rms_error_vs_original": relative_rms_error[variant],
                        "timer_scope": "same_process_safe_ABBA_serial_checked_typed_factor_iq_table_storage",
                    })
                );
                assert!(relative_rms_error[variant] <= 2e-5, "IQ grouped32 component relative RMS kind={kind}");
                if variant == 0 {
                    assert!(bitwise_equal[variant], "IQ control component changed Safe bits kind={kind}");
                }
            }
        }
    }
}
