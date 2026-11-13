// SPDX-License-Identifier: Apache-2.0
//! Test-only Ripple K-quant component experiments. No runtime policy changes.
use stark_engine_macros::stark_test;

use super::*;

fn pipeline(
    kind: u32,
    name: &str,
) -> PipelineState {
    let constants = MTLFunctionConstantValues::new();
    for (index, value) in [(0, kind), (4, 256), (5, weights::block_layout(kind).unwrap().1 as u32), (73, 0)] {
        constants.set_constant_value_type_at_index(NonNull::from(&value).cast(), MTLDataType::UInt, index);
    }
    let executor = executor().unwrap();
    let function = executor.library.new_function_with_name_constant_values(name, &constants).unwrap();
    executor.device.new_compute_pipeline_state_with_function(&function).unwrap()
}

fn codes(
    kind: u32,
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let (_, block_bytes) = weights::block_layout(kind).unwrap();
    let mut state = 0x9135_acedu32;
    let mut codes = vec![0u8; rows * columns / 256 * block_bytes];
    for byte in &mut codes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    for (i, block) in codes.chunks_exact_mut(block_bytes).enumerate() {
        let scale = half::f16::from_f32(0.001 + (i % 257) as f32 * 0.00003125).to_bits().to_le_bytes();
        if kind == 12 {
            block[..2].copy_from_slice(&scale);
            block[2..4]
                .copy_from_slice(&half::f16::from_f32(0.0004 + (i % 131) as f32 * 0.000015625).to_bits().to_le_bytes());
        } else {
            block[208..210].copy_from_slice(&scale);
        }
    }
    codes
}

fn oracle(
    kind: u32,
    encoded: &[u8],
    input: &[f32],
) -> (f64, f64) {
    let (_, bytes) = weights::block_layout(kind).unwrap();
    let mut expected = 0.0;
    let mut absolute = 0.0;
    for (block, x) in encoded.chunks_exact(bytes).zip(input.chunks_exact(256)) {
        let mut decoded = [0.0; 256];
        weights::decode_block(kind, block, &mut decoded).unwrap();
        for (&w, &x) in decoded.iter().zip(x) {
            let p = f64::from(w) * f64::from(x);
            expected += p;
            absolute += p.abs();
        }
    }
    (expected, absolute)
}

#[stark_test]
#[ignore]
fn ripple_k_quant_gpu_component_timings() {
    component_timings(false, false);
}

#[stark_test]
#[ignore]
fn ripple_q4_rowreuse_gpu_component_timings() {
    component_timings(true, false);
}

#[stark_test]
#[ignore]
fn ripple_q6_head_gpu_component_timings() {
    component_timings(false, true);
}

fn component_timings(
    row_reuse: bool,
    q6_head_only: bool,
) {
    let _activity = super::super::activity::InferenceActivity::begin().unwrap();
    let executor = executor().unwrap();
    for (kind, label, rows, columns) in [
        (12u32, "q4_gate_up", 40960usize, 5120usize),
        (12, "q4_down", 5120, 40960),
        (12, "q4_q_gate", 10240, 5120),
        (12, "q4_k_v", 1024, 5120),
        (12, "q4_o", 5120, 10240),
        (12, "q4_head", 100352, 5120),
        (14, "q6_down", 5120, 40960),
        (14, "q6_k_v", 1024, 5120),
        (14, "q6_head", 100352, 5120),
    ] {
        if q6_head_only && label != "q6_head" {
            continue;
        }
        if row_reuse && kind != 12 {
            continue;
        }
        let names = if row_reuse {
            [
                "hybrid_q4_grouped32_matvec",
                "hybrid_ripple_q4_grouped16_r2_matvec",
                "hybrid_ripple_q4_grouped32_r2_matvec",
            ]
        } else if kind == 12 {
            ["hybrid_packed_block_matvec", "hybrid_packed_factored_matvec", "hybrid_q4_grouped32_matvec"]
        } else {
            ["hybrid_packed_block_matvec", "hybrid_packed_factored_matvec", "hybrid_q6_grouped16_matvec"]
        };
        let pipelines = names.map(|name| pipeline(kind, name));
        let block_bytes = weights::block_layout(kind).unwrap().1;
        let matrix_bytes = rows * columns / 256 * block_bytes;
        let row_bytes = matrix_bytes / rows;
        let banks = (320usize * 1024 * 1024).div_ceil(matrix_bytes).max(2);
        let encoded = codes(kind, rows * banks, columns);
        let values = (0..columns).map(|i| (i as f32 * 0.017).sin()).collect::<Vec<_>>();
        let mut expected = Vec::new();
        for bank in 0..banks {
            for row in [0, rows / 3, rows * 2 / 3, rows - 1] {
                let at = bank * matrix_bytes + row * row_bytes;
                expected.push((bank * rows + row, oracle(kind, &encoded[at..at + row_bytes], &values)));
            }
        }
        let encoded = buffer(executor, &encoded).unwrap();
        let input = buffer(executor, bytemuck::cast_slice(&values)).unwrap();
        let output = executor.device.new_buffer(rows * banks * 4, MTLResourceOptions::STORAGE_MODE_SHARED).unwrap();
        let parameters = [rows as u32, columns as u32, kind, 256, block_bytes as u32, 0, 0, 0];
        let mut timings = vec![vec![Vec::<f64>::new(); 2]; 2];
        let mut reference = Vec::new();
        let mut max_error = [0.0f64; 3];
        let mut relative_rms = [0.0f64; 3];
        let mut normalized_f64 = [0.0f64; 3];
        for replay in 0..7 {
            for candidate in if replay % 2 == 0 {
                [1usize, 2]
            } else {
                [2usize, 1]
            } {
                for variant in [0, candidate, candidate, 0] {
                    let (result, seconds) = objc2::rc::autoreleasepool(|_| {
                        let command = executor.queue.command_buffer().unwrap();
                        let encoding = command.compute_command_encoder().unwrap();
                        encoding.set_compute_pipeline_state(&pipelines[variant]);
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
                                    width: rows.div_ceil(
                                        if (row_reuse && variant > 0) || (kind == 14 && variant == 2) {
                                            4
                                        } else {
                                            2
                                        },
                                    ),
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
                        // SAFETY: the owned shared outputs have completed their sole writers.
                        let result = unsafe {
                            std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), rows * banks)
                        }
                        .to_vec();
                        (result, (command.gpu_end_time() - command.gpu_start_time()) / banks as f64)
                    });
                    assert!(result.iter().all(|v| v.is_finite()));
                    if reference.is_empty() {
                        reference = result.clone();
                    }
                    if replay > 0 {
                        timings[candidate - 1][usize::from(variant != 0)].push(seconds);
                    }
                    let squared_error: f64 =
                        result.iter().zip(&reference).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
                    let squared_reference: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    relative_rms[variant] =
                        relative_rms[variant].max((squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt());
                    max_error[variant] = max_error[variant].max(
                        result
                            .iter()
                            .zip(&reference)
                            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                            .fold(0.0, f64::max),
                    );
                    for &(row, (value, absolute)) in &expected {
                        normalized_f64[variant] =
                            normalized_f64[variant].max((f64::from(result[row]) - value).abs() / absolute.max(1.0));
                    }
                    assert!(relative_rms[variant] <= 2e-5);
                    assert!(normalized_f64[variant] <= 2e-6);
                }
            }
        }
        for (pair, schedules) in timings.iter_mut().enumerate() {
            for (which, samples) in schedules.iter_mut().enumerate() {
                samples.sort_by(f64::total_cmp);
                let variant = if which == 0 {
                    0
                } else {
                    pair + 1
                };
                let seconds = (samples[5] + samples[6]) * 0.5;
                println!(
                    "{}",
                    serde_json::json!({"component":label,"rows":rows,"columns":columns,"kind":kind,"schedule":names[variant],"pair_candidate":names[pair+1],"math_mode":"safe","timer_scope":"serial_ABBA_rotating_banks_GPU_timestamps","encoded_population_bytes":matrix_bytes*banks,"rotating_banks":banks,"warm_samples":samples.len(),"median_gpu_microseconds_per_matrix":seconds*1e6,"encoded_gigabytes_per_second":matrix_bytes as f64/seconds/1e9,"maximum_absolute_error":max_error[variant],"relative_rms_error":relative_rms[variant],"maximum_f64_error_over_absolute_sum":normalized_f64[variant]})
                );
            }
        }
    }
}
