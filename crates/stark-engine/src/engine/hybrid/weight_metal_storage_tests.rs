// SPDX-License-Identifier: Apache-2.0
//! Safe packed GEMV controls with identical Shared/Private encoded weights.
use metal::MTLBlitCommandEncoder;
use stark_engine_macros::stark_test;

use super::*;

#[derive(Clone, Copy)]
struct HazardTiming {
    gpu: f64,
    encode: f64,
    commit: f64,
    wait: f64,
    roundtrip: f64,
}

fn immutable_shared(
    bytes: &[u8],
    untracked: bool,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let options = MTLResourceOptions::STORAGE_MODE_SHARED
        | if untracked {
            MTLResourceOptions::HAZARD_TRACKING_MODE_UNTRACKED
        } else {
            MTLResourceOptions::empty()
        };
    executor().unwrap().device.new_buffer_with_data(bytes, options).unwrap()
}

fn run_hazard_population(
    pipeline: &PipelineState,
    weights: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    p: [u32; 8],
    rotation: usize,
) -> (Vec<f32>, HazardTiming) {
    objc2::rc::autoreleasepool(|_| {
        // Weights are immutable after synchronous construction. There are no
        // GPU writes to any untracked resource; only inputs/outputs are shared
        // across dispatches, and they retain default tracked hazard handling.
        // The borrowed owners remain live until the completion fence below.
        let start = std::time::Instant::now();
        let executor = executor().unwrap();
        let command = executor.queue.command_buffer().unwrap();
        let encoding = command.compute_command_encoder().unwrap();
        encoding.set_compute_pipeline_state(pipeline);
        encoding.set_buffer(Some(input), 0, 1);
        encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        let offsets = [0u32; 4];
        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
        let rows = p[0] as usize;
        for i in 0..weights.len() {
            let bank = (i + rotation) % weights.len();
            encoding.set_buffer(Some(&weights[bank]), 0, 0);
            encoding.set_buffer(Some(output), bank * rows * 4, 2);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: rows.div_ceil(if p[2] == 13 {
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
        }
        encoding.end_encoding();
        let encode = start.elapsed().as_secs_f64();
        let mark = std::time::Instant::now();
        command.commit();
        let commit = mark.elapsed().as_secs_f64();
        let mark = std::time::Instant::now();
        command.wait_until_completed();
        let wait = mark.elapsed().as_secs_f64();
        let roundtrip = start.elapsed().as_secs_f64();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let timing = HazardTiming {
            gpu: command.gpu_end_time() - command.gpu_start_time(),
            encode,
            commit,
            wait,
            roundtrip,
        };
        // SAFETY: only the tracked Shared output is accessed on the CPU, after
        // all GPU writers complete. Untracked weights are never GPU-written.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * weights.len()) }
                .to_vec();
        assert!(values.iter().all(|v| v.is_finite()));
        (values, timing)
    })
}

#[stark_test]
fn immutable_untracked_weights_match_tracked_bits_and_f64() {
    let executor = executor().unwrap();
    for kind in [13u32, 14] {
        let (rows, columns) = (13usize, 768usize);
        let (_, bytes) = weights::block_layout(kind).unwrap();
        let encoded = codes(kind, rows, columns);
        let tracked = vec![immutable_shared(&encoded, false)];
        let untracked = vec![immutable_shared(&encoded, true)];
        let x = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let input = buffer(executor, bytemuck::cast_slice(&x)).unwrap();
        let output = executor.device.new_buffer(rows * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let p = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, 0, 0];
        let pipeline = pipeline(kind);
        let (a, _) = run_hazard_population(&pipeline, &tracked, &input, &output, p, 0);
        let (b, _) = run_hazard_population(&pipeline, &untracked, &input, &output, p, 0);
        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
        for (row, (expected, absolute)) in f64_oracle(kind, &encoded, rows, columns, &x, 0).into_iter().enumerate() {
            assert!((f64::from(b[row]) - expected).abs() <= 2e-6 * absolute.max(1.0));
        }
    }
}

#[stark_test]
#[ignore]
fn immutable_weight_hazard_tracking_gpu_and_cpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let cases = [
        ("glm_q5k_main", 13u32, 8192usize, 4096usize, 0usize),
        ("glm_q5k_output", 13, 4096, 8192, 0),
        ("glm_q6k_main", 14, 8192, 4096, 0),
        ("glm_q6k_output", 14, 4096, 8192, 0),
        ("q5k_600_unique_bindings", 13, 512, 4096, 600),
    ];
    for (label, kind, rows, columns, count) in cases {
        objc2::rc::autoreleasepool(|_| {
            let executor = executor().unwrap();
            let (_, bytes) = weights::block_layout(kind).unwrap();
            let matrix_bytes = rows * columns / 256 * bytes;
            let count = if count == 0 {
                (160usize * 1024 * 1024).div_ceil(matrix_bytes)
            } else {
                count
            };
            // Distinct owning Metal buffers model resource-tracking cardinality.
            // Replicas use the same encoded matrix to keep allocation preparation
            // bounded; every physical buffer remains live through every wait.
            let encoded = codes(kind, rows, columns);
            let populations = [
                (0..count).map(|_| immutable_shared(&encoded, false)).collect::<Vec<_>>(),
                (0..count).map(|_| immutable_shared(&encoded, true)).collect::<Vec<_>>(),
            ];
            for population in &populations {
                assert_eq!(
                    population
                        .iter()
                        .map(|b| Retained::as_ptr(b) as usize)
                        .collect::<std::collections::HashSet<_>>()
                        .len(),
                    count
                );
            }
            let x = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let mut oracle = Vec::new();
            for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                let at = row * matrix_bytes / rows;
                oracle.push((row, f64_oracle(kind, &encoded[at..at + matrix_bytes / rows], 1, columns, &x, 0)[0]));
            }
            let input = buffer(executor, bytemuck::cast_slice(&x)).unwrap();
            let output = executor.device.new_buffer(rows * count * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
            let pipeline = pipeline(kind);
            let p = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, 0, 0];
            let mut samples = [Vec::new(), Vec::new()];
            let mut reference = Vec::<f32>::new();
            let mut bitwise = [true; 2];
            let mut error = [0.0f64; 2];
            for replay in 0..7 {
                for variant in if replay % 2 == 0 {
                    [0usize, 1, 1, 0]
                } else {
                    [1usize, 0, 0, 1]
                } {
                    let (values, timing) =
                        run_hazard_population(&pipeline, &populations[variant], &input, &output, p, replay);
                    if reference.is_empty() {
                        reference = values.clone();
                    }
                    if replay > 0 {
                        samples[variant].push(timing);
                    }
                    bitwise[variant] &= values.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits());
                    for bank in 0..count {
                        for &(row, (expected, absolute)) in &oracle {
                            error[variant] = error[variant]
                                .max((f64::from(values[bank * rows + row]) - expected).abs() / absolute.max(1.0));
                        }
                    }
                }
            }
            for variant in 0..2 {
                let median = |field: fn(&HazardTiming) -> f64| {
                    let mut v = samples[variant].iter().map(field).collect::<Vec<_>>();
                    v.sort_by(f64::total_cmp);
                    (v[5] + v[6]) * 0.5
                };
                println!(
                    "{}",
                    serde_json::json!({"component":label,"kind":kind,"rows":rows,"columns":columns,"weight_hazard_tracking":if variant==0{"default_tracked"}else{"untracked"},"activity_assertion":true,"math_mode":"safe","unique_weight_buffers":count,"encoded_population_bytes":matrix_bytes*count,"identical_matrix_replicas":true,"inputs_outputs_default_tracked":true,"immutable_weights_never_gpu_written":true,"warm_samples":samples[variant].len(),"median_gpu_microseconds_per_binding":median(|t|t.gpu)*1e6/count as f64,"median_cpu_encode_microseconds_per_command":median(|t|t.encode)*1e6,"median_cpu_commit_microseconds_per_command":median(|t|t.commit)*1e6,"median_cpu_wait_microseconds_per_command":median(|t|t.wait)*1e6,"median_cpu_roundtrip_microseconds_per_command":median(|t|t.roundtrip)*1e6,"bitwise_equal_to_tracked":bitwise[variant],"maximum_f64_error_over_absolute_sum":error[variant],"cpu_timer_scope":"command_create_encode_commit_wait_excludes_allocations_upload_validation_readback","gpu_timer_scope":"same_process_ABBA_unique_buffers_gpu_timestamps"})
                );
                assert!(bitwise[variant]);
                assert!(error[variant] <= 2e-6);
            }
        });
    }
}

fn pipeline(kind: u32) -> PipelineState {
    let executor = executor().unwrap();
    let (_, bytes) = weights::block_layout(kind).unwrap();
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, bytes as u32), (73, 0)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let name = if kind == 13 {
        "hybrid_q5_grouped32_matvec"
    } else {
        "hybrid_q6_grouped16_matvec"
    };
    let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

fn private_copy(shared: &ProtocolObject<dyn MTLBuffer>) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let executor = executor().unwrap();
    let private = executor.device.new_buffer(shared.length(), MTLResourceOptions::STORAGE_MODE_PRIVATE).unwrap();
    let command = executor.queue.command_buffer().unwrap();
    let blit = command.blit_command_encoder().unwrap();
    blit.copy_buffer_to_buffer(shared, 0, &private, 0, shared.length());
    blit.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    private
}

#[expect(clippy::too_many_arguments, reason = "test keeps identical storage/bank/offset geometry explicit")]
fn run(
    pipeline: &PipelineState,
    weights: &ProtocolObject<dyn MTLBuffer>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    parameters: [u32; 8],
    banks: usize,
    prefix: usize,
    rotation: usize,
) -> (Vec<f32>, f64) {
    objc2::rc::autoreleasepool(|_| {
        let executor = executor().unwrap();
        let rows = parameters[0] as usize;
        let columns = parameters[1] as usize;
        let matrix_bytes = rows * columns / 256 * parameters[4] as usize;
        let command = executor.queue.command_buffer().unwrap();
        let encoding = command.compute_command_encoder().unwrap();
        encoding.set_compute_pipeline_state(pipeline);
        encoding.set_buffer(Some(input), 0, 1);
        encoding.set_bytes(NonNull::from(&parameters).cast(), std::mem::size_of_val(&parameters), 3);
        let offsets = [0u32; 4];
        encoding.set_bytes(NonNull::from(&offsets).cast(), std::mem::size_of_val(&offsets), 6);
        for i in 0..banks {
            let bank = (i + rotation) % banks;
            encoding.set_buffer(Some(weights), prefix + bank * matrix_bytes, 0);
            encoding.set_buffer(Some(output), bank * rows * 4, 2);
            encoding.dispatch_threadgroups(
                MTLSize {
                    width: rows.div_ceil(if parameters[2] == 13 {
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
        }
        encoding.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: only the Shared output is read after its writers complete.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * banks) }.to_vec();
        assert!(values.iter().all(|v| v.is_finite()));
        (values, (command.gpu_end_time() - command.gpu_start_time()) / banks as f64)
    })
}

#[stark_test]
fn shared_private_packed_weights_preserve_safe_bits_offsets_tails_and_f64() {
    let executor = executor().unwrap();
    for kind in [13u32, 14] {
        let (rows, columns, prefix) = (13usize, 768usize, 64usize);
        let (_, bytes) = weights::block_layout(kind).unwrap();
        let encoded = codes(kind, rows, columns);
        let mut storage = vec![0xa5; prefix];
        storage.extend_from_slice(&encoded);
        storage.extend_from_slice(&[0x5a; 64]);
        let shared = buffer(executor, &storage).unwrap();
        let private = private_copy(&shared);
        let pipeline = pipeline(kind);
        let x = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let input = buffer(executor, bytemuck::cast_slice(&x)).unwrap();
        let output = executor.device.new_buffer(rows * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let p = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, 0, 0];
        let (a, _) = run(&pipeline, &shared, &input, &output, p, 1, prefix, 0);
        let (b, _) = run(&pipeline, &private, &input, &output, p, 1, prefix, 0);
        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
        for (row, (expected, absolute)) in f64_oracle(kind, &encoded, rows, columns, &x, 0).into_iter().enumerate() {
            assert!((f64::from(b[row]) - expected).abs() <= 2e-6 * absolute.max(1.0));
        }
    }
}

#[stark_test]
#[ignore]
fn shared_private_packed_weights_gpu_component_timings() {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let cases = [
        ("glm_q5k_main", 13u32, 8192usize, 4096usize),
        ("glm_q5k_output", 13, 4096, 8192),
        ("glm_q6k_main", 14, 8192, 4096),
        ("glm_q6k_output", 14, 4096, 8192),
    ];
    for (label, kind, rows, columns) in cases {
        objc2::rc::autoreleasepool(|_| {
            let executor = executor().unwrap();
            let (_, bytes) = weights::block_layout(kind).unwrap();
            let matrix_bytes = rows * columns / 256 * bytes;
            let banks = (160usize * 1024 * 1024).div_ceil(matrix_bytes);
            let encoded = codes(kind, rows * banks, columns);
            let x = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
            let mut oracle = Vec::new();
            for bank in 0..banks {
                for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                    let at = bank * matrix_bytes + row * matrix_bytes / rows;
                    oracle.push((
                        bank * rows + row,
                        f64_oracle(kind, &encoded[at..at + matrix_bytes / rows], 1, columns, &x, 0)[0],
                    ));
                }
            }
            let shared = buffer(executor, &encoded).unwrap();
            let private = private_copy(&shared);
            drop(encoded);
            let pipeline = pipeline(kind);
            let input = buffer(executor, bytemuck::cast_slice(&x)).unwrap();
            let output = executor.device.new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
            let p = [rows as u32, columns as u32, kind, 256, bytes as u32, 0, 0, 0];
            let mut samples = [Vec::new(), Vec::new()];
            let mut reference = Vec::<f32>::new();
            let mut bitwise = [true; 2];
            let mut f64_error = [0.0f64; 2];
            for replay in 0..7 {
                for variant in if replay % 2 == 0 {
                    [0usize, 1, 1, 0]
                } else {
                    [1usize, 0, 0, 1]
                } {
                    let source = if variant == 0 {
                        &shared
                    } else {
                        &private
                    };
                    let (values, seconds) = run(&pipeline, source, &input, &output, p, banks, 0, replay);
                    if reference.is_empty() {
                        reference = values.clone();
                    }
                    if replay > 0 {
                        samples[variant].push(seconds);
                    }
                    bitwise[variant] &= values.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits());
                    for &(row, (expected, absolute)) in &oracle {
                        f64_error[variant] =
                            f64_error[variant].max((f64::from(values[row]) - expected).abs() / absolute.max(1.0));
                    }
                }
            }
            for (variant, samples) in samples.iter_mut().enumerate() {
                samples.sort_by(f64::total_cmp);
                let seconds = (samples[5] + samples[6]) * 0.5;
                println!(
                    "{}",
                    serde_json::json!({"component":label,"kind":kind,"rows":rows,"columns":columns,"storage":if variant==0{"shared"}else{"private"},"activity_assertion":true,"math_mode":"safe","rotating_banks":banks,"encoded_population_bytes":matrix_bytes*banks,"private_upload_fenced_before_timing":true,"weight_offset_prefix":0,"warm_samples":samples.len(),"median_gpu_microseconds_per_matrix":seconds*1e6,"minimum_gpu_microseconds_per_matrix":samples[0]*1e6,"maximum_gpu_microseconds_per_matrix":samples[samples.len()-1]*1e6,"encoded_gigabytes_per_second":matrix_bytes as f64/seconds/1e9,"bitwise_equal_to_shared":bitwise[variant],"maximum_f64_error_over_absolute_sum":f64_error[variant],"timer_scope":"same_process_ABBA_serial_rotating_banks_gpu_timestamps_upload_excluded"})
                );
                assert!(bitwise[variant]);
                assert!(f64_error[variant] <= 2e-6);
            }
        });
    }
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
